"""Custom agent implementation using StateGraph."""

from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.checkpoint.memory import MemorySaver

from ..models import ModelConfig, create_chat_model
from ..tools import get_tools_by_names


class AgentState(TypedDict):
    """State for custom agent."""
    messages: List[BaseMessage]
    current_step: int
    max_steps: int
    should_continue: bool
    metadata: Dict[str, Any]


def create_custom_graph_agent(
    model_config: ModelConfig,
    tools: List[str],
    max_steps: int = 10,
    include_human_approval: bool = False,
    custom_nodes: Optional[Dict[str, Any]] = None
) -> CompiledStateGraph:
    """Create a custom agent using StateGraph.
    
    Args:
        model_config: Model configuration
        tools: List of tool names
        max_steps: Maximum steps before stopping
        include_human_approval: Whether to include human approval step
        custom_nodes: Optional custom nodes to add to the graph
        
    Returns:
        Compiled custom agent graph
    """
    # Create model
    model = create_chat_model(model_config)
    
    # Get tools
    tool_instances = get_tools_by_names(tools)
    
    # Create tool node
    tool_node = ToolNode(tool_instances)
    
    # Define nodes
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Call the language model."""
        messages = state["messages"]
        current_step = state.get("current_step", 0)
        
        # Add step tracking to prompt
        if current_step > 0:
            step_message = f"[Step {current_step}/{max_steps}]"
            if messages and isinstance(messages[-1], HumanMessage):
                messages[-1].content = f"{step_message} {messages[-1].content}"
        
        # Bind tools to model
        model_with_tools = model.bind_tools(tool_instances)
        
        # Call model
        response = model_with_tools.invoke(messages)
        
        # Update state
        return {
            "messages": [response],
            "current_step": current_step + 1
        }
    
    def should_continue(state: AgentState) -> str:
        """Determine if we should continue or stop."""
        messages = state["messages"]
        current_step = state.get("current_step", 0)
        
        # Check if we've exceeded max steps
        if current_step >= max_steps:
            return "end"
        
        # Check last message for tool calls
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Check for specific stop conditions
        if isinstance(last_message, AIMessage):
            content = last_message.content.lower()
            if any(phrase in content for phrase in ["task completed", "done", "finished"]):
                return "end"
        
        return "end"
    
    def human_approval(state: AgentState) -> Dict[str, Any]:
        """Placeholder for human approval step."""
        # In a real implementation, this would pause and wait for human input
        print("🔔 Human approval required. Auto-approving for demo...")
        return {"metadata": {"human_approved": True}}
    
    def process_tools(state: AgentState) -> Dict[str, Any]:
        """Process tool calls."""
        return tool_node.invoke(state)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", process_tools)
    
    if include_human_approval:
        workflow.add_node("human_approval", human_approval)
    
    # Add custom nodes if provided
    if custom_nodes:
        for node_name, node_func in custom_nodes.items():
            workflow.add_node(node_name, node_func)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    if include_human_approval:
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "human_approval",
                "end": END
            }
        )
        workflow.add_edge("human_approval", "tools")
    else:
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
    
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def create_stateful_tool_agent(
    model_config: ModelConfig,
    stateful_tools: List[BaseTool]
) -> CompiledStateGraph:
    """Create an agent with tools that can access and modify state.
    
    Args:
        model_config: Model configuration
        stateful_tools: List of tools with state injection
        
    Returns:
        Compiled agent with stateful tools
    """
    from langchain_core.tools import tool
    
    # Example of creating a stateful tool
    @tool
    def update_context(
        key: str,
        value: str,
        state: Annotated[dict, InjectedState]
    ) -> str:
        """Update context in agent state."""
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"][key] = value
        return f"Updated context: {key} = {value}"
    
    @tool
    def get_context(
        key: str,
        state: Annotated[dict, InjectedState]
    ) -> str:
        """Get context from agent state."""
        metadata = state.get("metadata", {})
        value = metadata.get(key, "Not found")
        return f"Context value for {key}: {value}"
    
    # Combine with provided stateful tools
    all_tools = stateful_tools + [update_context, get_context]
    
    # Create model
    model = create_chat_model(model_config)
    
    # Create tool node that handles state injection
    tool_node = ToolNode(all_tools)
    
    # Define the agent logic
    def agent(state: AgentState) -> Dict[str, Any]:
        """Agent that works with stateful tools."""
        messages = state["messages"]
        model_with_tools = model.bind_tools(all_tools)
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def route(state: AgentState) -> str:
        """Route based on tool calls."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()