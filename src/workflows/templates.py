"""Pre-built workflow templates."""

from typing import List, Dict, Any, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .workflow_builder import WorkflowBuilder, WorkflowConfig, WorkflowState
from ..models import ModelConfig, create_chat_model
from ..agents import create_react_agent_wrapper, ReactAgentConfig


def create_sequential_workflow(
    steps: List[Dict[str, Any]],
    name: str = "sequential_workflow",
    enable_memory: bool = False
) -> Any:
    """Create a sequential workflow where steps execute one after another.
    
    Args:
        steps: List of step configurations with 'name' and 'func' keys
        name: Workflow name
        enable_memory: Whether to enable memory
        
    Returns:
        Compiled workflow
        
    Example:
        >>> steps = [
        ...     {"name": "step1", "func": lambda s: s},
        ...     {"name": "step2", "func": lambda s: s},
        ...     {"name": "step3", "func": lambda s: s}
        ... ]
        >>> workflow = create_sequential_workflow(steps)
    """
    config = WorkflowConfig(
        name=name,
        enable_memory=enable_memory
    )
    
    builder = WorkflowBuilder(config)
    
    # Add nodes
    for step in steps:
        builder.add_node(
            name=step["name"],
            func=step["func"],
            description=step.get("description", "")
        )
    
    # Add edges to create sequential flow
    for i in range(len(steps)):
        if i == 0:
            builder.add_edge("START", steps[i]["name"])
        else:
            builder.add_edge(steps[i-1]["name"], steps[i]["name"])
        
        if i == len(steps) - 1:
            builder.add_edge(steps[i]["name"], "END")
    
    return builder.compile()


def create_parallel_workflow(
    parallel_steps: List[Dict[str, Any]],
    aggregator_func: Optional[Callable] = None,
    name: str = "parallel_workflow",
    enable_memory: bool = False
) -> Any:
    """Create a workflow where steps execute in parallel.
    
    Args:
        parallel_steps: List of step configurations
        aggregator_func: Function to aggregate results
        name: Workflow name
        enable_memory: Whether to enable memory
        
    Returns:
        Compiled workflow
    """
    config = WorkflowConfig(
        name=name,
        enable_memory=enable_memory
    )
    
    builder = WorkflowBuilder(config)
    
    # Default aggregator
    if not aggregator_func:
        def default_aggregator(state: WorkflowState) -> WorkflowState:
            """Default aggregator that combines messages."""
            return state
        aggregator_func = default_aggregator
    
    # Add parallel nodes
    for step in parallel_steps:
        builder.add_node(
            name=step["name"],
            func=step["func"],
            description=step.get("description", "")
        )
        builder.add_edge("START", step["name"])
    
    # Add aggregator
    builder.add_node("aggregator", aggregator_func, "Aggregate results")
    
    # Connect all parallel nodes to aggregator
    for step in parallel_steps:
        builder.add_edge(step["name"], "aggregator")
    
    builder.add_edge("aggregator", "END")
    
    return builder.compile()


def create_conditional_workflow(
    initial_step: Dict[str, Any],
    condition_func: Callable,
    branches: Dict[str, List[Dict[str, Any]]],
    name: str = "conditional_workflow",
    enable_memory: bool = False
) -> Any:
    """Create a workflow with conditional branching.
    
    Args:
        initial_step: Initial step configuration
        condition_func: Function to determine branch
        branches: Dictionary mapping condition results to step lists
        name: Workflow name
        enable_memory: Whether to enable memory
        
    Returns:
        Compiled workflow
        
    Example:
        >>> def check_condition(state):
        ...     return "branch_a" if some_condition else "branch_b"
        >>> 
        >>> branches = {
        ...     "branch_a": [{"name": "a1", "func": func_a1}],
        ...     "branch_b": [{"name": "b1", "func": func_b1}]
        ... }
        >>> workflow = create_conditional_workflow(
        ...     initial_step={"name": "start", "func": start_func},
        ...     condition_func=check_condition,
        ...     branches=branches
        ... )
    """
    config = WorkflowConfig(
        name=name,
        enable_memory=enable_memory
    )
    
    builder = WorkflowBuilder(config)
    
    # Add initial step
    builder.add_node(
        name=initial_step["name"],
        func=initial_step["func"],
        description=initial_step.get("description", "")
    )
    builder.add_edge("START", initial_step["name"])
    
    # Add branch nodes
    branch_entry_points = {}
    for branch_name, branch_steps in branches.items():
        for i, step in enumerate(branch_steps):
            builder.add_node(
                name=step["name"],
                func=step["func"],
                description=step.get("description", "")
            )
            
            # Track entry point for each branch
            if i == 0:
                branch_entry_points[branch_name] = step["name"]
            
            # Connect steps within branch
            if i > 0:
                builder.add_edge(branch_steps[i-1]["name"], step["name"])
            
            # Connect last step to END
            if i == len(branch_steps) - 1:
                builder.add_edge(step["name"], "END")
    
    # Add conditional routing
    routes = {
        branch_name: entry_point
        for branch_name, entry_point in branch_entry_points.items()
    }
    
    # Add default route to END if no branches match
    routes["default"] = "END"
    
    builder.add_conditional_edge(
        from_node=initial_step["name"],
        condition=condition_func,
        routes=routes
    )
    
    return builder.compile()


def create_human_in_loop_workflow(
    agent_config: ReactAgentConfig,
    approval_required_for: List[str] = None,
    name: str = "human_in_loop_workflow",
    enable_memory: bool = True
) -> Any:
    """Create a workflow with human approval steps.
    
    Args:
        agent_config: Configuration for the agent
        approval_required_for: List of tool names requiring approval
        name: Workflow name
        enable_memory: Whether to enable memory
        
    Returns:
        Compiled workflow with human approval
    """
    if approval_required_for is None:
        approval_required_for = ["write_file", "web_search"]
    
    config = WorkflowConfig(
        name=name,
        enable_memory=enable_memory,
        description="Workflow with human approval for sensitive operations"
    )
    
    builder = WorkflowBuilder(config)
    
    # Create agent
    agent = create_react_agent_wrapper(agent_config)
    
    # Define agent node
    def agent_node(state: WorkflowState) -> WorkflowState:
        """Run the agent."""
        messages = state.get("messages", [])
        
        # Invoke agent
        result = agent.invoke({"messages": messages})
        
        # Update state
        state["messages"] = result.get("messages", messages)
        
        # Check if approval is needed
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                if tool_call["name"] in approval_required_for:
                    state["metadata"]["needs_approval"] = True
                    state["metadata"]["pending_tool"] = tool_call["name"]
                    break
        
        return state
    
    # Define human approval node
    def human_approval(state: WorkflowState) -> WorkflowState:
        """Request human approval."""
        pending_tool = state["metadata"].get("pending_tool", "unknown")
        
        # In a real implementation, this would pause and wait for input
        print(f"🔔 Human approval required for tool: {pending_tool}")
        print("Auto-approving for demo...")
        
        # Simulate approval
        approval_message = AIMessage(
            content=f"Human approved use of {pending_tool}. Proceeding..."
        )
        state["messages"].append(approval_message)
        state["metadata"]["needs_approval"] = False
        state["metadata"]["approved"] = True
        
        return state
    
    # Define tool execution node
    def execute_tools(state: WorkflowState) -> WorkflowState:
        """Execute approved tools."""
        # Tool execution would happen here
        return state
    
    # Define routing logic
    def route_agent(state: WorkflowState) -> str:
        """Route based on agent output."""
        if state.get("metadata", {}).get("needs_approval"):
            return "human_approval"
        
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
        
        return "END"
    
    # Build the workflow
    builder.add_node("agent", agent_node, "Main agent")
    builder.add_node("human_approval", human_approval, "Human approval step")
    builder.add_node("tools", execute_tools, "Tool execution")
    
    builder.add_edge("START", "agent")
    builder.add_conditional_edge(
        "agent",
        route_agent,
        {
            "human_approval": "human_approval",
            "tools": "tools",
            "END": "END"
        }
    )
    builder.add_edge("human_approval", "tools")
    builder.add_edge("tools", "agent")
    
    return builder.compile()