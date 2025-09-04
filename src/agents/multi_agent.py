"""Multi-agent system implementation with supervisor pattern."""

from typing import Dict, List, Optional, Any, Literal
from typing_extensions import TypedDict
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledStateGraph

from ..models import ModelConfig, create_chat_model
from ..tools import get_tools_by_names


class MessagesState(TypedDict):
    """State for multi-agent system."""
    messages: List[BaseMessage]
    next: Optional[str]


@dataclass
class AgentConfig:
    """Configuration for individual agent."""
    name: str
    model_config: ModelConfig
    tools: List[str]
    prompt: str
    description: str  # For handoff tool


def create_handoff_tool(agent_name: str, description: str):
    """Create a handoff tool for transferring to another agent.
    
    Args:
        agent_name: Name of the agent to transfer to
        description: Description of when to use this handoff
        
    Returns:
        Tool for handoff
    """
    @tool(f"transfer_to_{agent_name}")
    def handoff(reason: str = "") -> str:
        """Transfer control to another agent."""
        return f"Transferring to {agent_name}: {reason}"
    
    handoff.description = description
    return handoff


def create_supervisor_agent(
    supervisor_config: ModelConfig,
    agents: Dict[str, AgentConfig],
    supervisor_prompt: Optional[str] = None
) -> CompiledStateGraph:
    """Create a supervisor-based multi-agent system.
    
    Args:
        supervisor_config: Model configuration for supervisor
        agents: Dictionary of agent configurations
        supervisor_prompt: Optional custom prompt for supervisor
        
    Returns:
        Compiled multi-agent graph
    """
    # Create handoff tools for supervisor
    handoff_tools = []
    for agent_name, agent_config in agents.items():
        tool = create_handoff_tool(agent_name, agent_config.description)
        handoff_tools.append(tool)
    
    # Create supervisor
    supervisor_model = create_chat_model(supervisor_config)
    
    if not supervisor_prompt:
        agent_descriptions = "\n".join([
            f"- {name}: {config.description}"
            for name, config in agents.items()
        ])
        supervisor_prompt = f"""You are a supervisor managing the following agents:
{agent_descriptions}

Assign work to the most appropriate agent based on the task.
Do not do any work yourself, only delegate."""
    
    supervisor = create_react_agent(
        model=supervisor_model,
        tools=handoff_tools,
        prompt=supervisor_prompt,
        name="supervisor"
    )
    
    # Create individual agents
    agent_nodes = {}
    for agent_name, agent_config in agents.items():
        model = create_chat_model(agent_config.model_config)
        tools = get_tools_by_names(agent_config.tools)
        
        # Add handoff tools to allow agents to transfer back
        other_agents = [name for name in agents.keys() if name != agent_name]
        for other_name in other_agents:
            other_config = agents[other_name]
            handoff = create_handoff_tool(other_name, other_config.description)
            tools.append(handoff)
        
        # Add supervisor handoff
        supervisor_handoff = create_handoff_tool("supervisor", "Transfer back to supervisor for task routing")
        tools.append(supervisor_handoff)
        
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=agent_config.prompt,
            name=agent_name
        )
        agent_nodes[agent_name] = agent
    
    # Build the graph
    workflow = StateGraph(MessagesState)
    
    # Add supervisor node
    workflow.add_node("supervisor", supervisor)
    
    # Add agent nodes
    for agent_name, agent in agent_nodes.items():
        workflow.add_node(agent_name, agent)
    
    # Define routing logic
    def route_supervisor(state: MessagesState) -> str:
        """Route from supervisor based on tool calls."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Check for handoff tool calls
            for tool_call in last_message.tool_calls:
                if tool_call["name"].startswith("transfer_to_"):
                    agent_name = tool_call["name"].replace("transfer_to_", "")
                    if agent_name in agents:
                        return agent_name
        
        return END
    
    def route_agent(state: MessagesState) -> str:
        """Route from agent based on tool calls."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Check for handoff tool calls
            for tool_call in last_message.tool_calls:
                if tool_call["name"].startswith("transfer_to_"):
                    target = tool_call["name"].replace("transfer_to_", "")
                    if target == "supervisor":
                        return "supervisor"
                    elif target in agents:
                        return target
        
        return END
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", route_supervisor)
    
    for agent_name in agents.keys():
        workflow.add_conditional_edges(agent_name, route_agent)
    
    return workflow.compile()


def create_multi_agent_system(
    agents: List[AgentConfig],
    orchestration_type: Literal["supervisor", "sequential", "parallel"] = "supervisor",
    supervisor_config: Optional[ModelConfig] = None
) -> CompiledStateGraph:
    """Create a multi-agent system with specified orchestration.
    
    Args:
        agents: List of agent configurations
        orchestration_type: Type of orchestration pattern
        supervisor_config: Model config for supervisor (if using supervisor pattern)
        
    Returns:
        Compiled multi-agent graph
    """
    if orchestration_type == "supervisor":
        if not supervisor_config:
            # Use default supervisor config
            from ..models import ModelProvider
            supervisor_config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4"
            )
        
        agents_dict = {agent.name: agent for agent in agents}
        return create_supervisor_agent(supervisor_config, agents_dict)
    
    elif orchestration_type == "sequential":
        # Create sequential workflow
        workflow = StateGraph(MessagesState)
        
        # Create and add agent nodes
        for i, agent_config in enumerate(agents):
            model = create_chat_model(agent_config.model_config)
            tools = get_tools_by_names(agent_config.tools)
            
            agent = create_react_agent(
                model=model,
                tools=tools,
                prompt=agent_config.prompt,
                name=agent_config.name
            )
            workflow.add_node(agent_config.name, agent)
            
            # Add edges
            if i == 0:
                workflow.add_edge(START, agent_config.name)
            else:
                workflow.add_edge(agents[i-1].name, agent_config.name)
            
            if i == len(agents) - 1:
                workflow.add_edge(agent_config.name, END)
        
        return workflow.compile()
    
    elif orchestration_type == "parallel":
        # Create parallel workflow (simplified version)
        workflow = StateGraph(MessagesState)
        
        # Add aggregator node
        def aggregate_responses(state: MessagesState) -> MessagesState:
            """Aggregate responses from all agents."""
            return state
        
        workflow.add_node("aggregator", aggregate_responses)
        
        # Create and add agent nodes
        for agent_config in agents:
            model = create_chat_model(agent_config.model_config)
            tools = get_tools_by_names(agent_config.tools)
            
            agent = create_react_agent(
                model=model,
                tools=tools,
                prompt=agent_config.prompt,
                name=agent_config.name
            )
            workflow.add_node(agent_config.name, agent)
            
            # Add edges for parallel execution
            workflow.add_edge(START, agent_config.name)
            workflow.add_edge(agent_config.name, "aggregator")
        
        workflow.add_edge("aggregator", END)
        
        return workflow.compile()
    
    else:
        raise ValueError(f"Unknown orchestration type: {orchestration_type}")