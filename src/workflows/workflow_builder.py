"""Workflow builder for creating complex agent workflows."""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.graph import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


class WorkflowState(TypedDict):
    """Base state for workflows."""
    messages: List[BaseMessage]
    current_node: str
    history: List[str]
    metadata: Dict[str, Any]
    error: Optional[str]


@dataclass
class NodeConfig:
    """Configuration for a workflow node."""
    name: str
    func: Callable
    description: str = ""
    retry_on_error: bool = False
    max_retries: int = 3
    timeout: Optional[int] = None


@dataclass
class EdgeConfig:
    """Configuration for workflow edges."""
    from_node: str
    to_node: Union[str, Callable]
    condition: Optional[Callable] = None
    label: str = ""


@dataclass
class WorkflowConfig:
    """Configuration for entire workflow."""
    name: str
    description: str = ""
    max_steps: int = 100
    enable_memory: bool = False
    memory_type: str = "in_memory"
    debug: bool = False
    visualize: bool = False


class WorkflowBuilder:
    """Builder for creating complex workflows."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize workflow builder.
        
        Args:
            config: Workflow configuration
        """
        self.config = config
        self.nodes: Dict[str, NodeConfig] = {}
        self.edges: List[EdgeConfig] = []
        self.state_graph = StateGraph(WorkflowState)
        self.entry_point: Optional[str] = None
        self.conditional_edges: Dict[str, Dict] = {}
    
    def add_node(
        self,
        name: str,
        func: Callable,
        description: str = "",
        retry_on_error: bool = False,
        max_retries: int = 3
    ) -> "WorkflowBuilder":
        """Add a node to the workflow.
        
        Args:
            name: Node name
            func: Node function
            description: Node description
            retry_on_error: Whether to retry on error
            max_retries: Maximum retry attempts
            
        Returns:
            Self for chaining
        """
        node_config = NodeConfig(
            name=name,
            func=func,
            description=description,
            retry_on_error=retry_on_error,
            max_retries=max_retries
        )
        
        # Wrap function with error handling if needed
        if retry_on_error:
            wrapped_func = self._wrap_with_retry(func, max_retries)
        else:
            wrapped_func = func
        
        self.nodes[name] = node_config
        self.state_graph.add_node(name, wrapped_func)
        
        # Set entry point if this is the first node
        if not self.entry_point:
            self.entry_point = name
        
        return self
    
    def add_edge(
        self,
        from_node: str,
        to_node: str,
        label: str = ""
    ) -> "WorkflowBuilder":
        """Add a simple edge between nodes.
        
        Args:
            from_node: Source node
            to_node: Destination node
            label: Edge label for visualization
            
        Returns:
            Self for chaining
        """
        edge_config = EdgeConfig(
            from_node=from_node,
            to_node=to_node,
            label=label
        )
        self.edges.append(edge_config)
        
        if from_node == "START":
            self.state_graph.add_edge(START, to_node)
            self.entry_point = to_node
        elif to_node == "END":
            self.state_graph.add_edge(from_node, END)
        else:
            self.state_graph.add_edge(from_node, to_node)
        
        return self
    
    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable,
        routes: Dict[str, str],
        label: str = ""
    ) -> "WorkflowBuilder":
        """Add conditional edge with routing logic.
        
        Args:
            from_node: Source node
            condition: Function to determine routing
            routes: Mapping of condition results to nodes
            label: Edge label
            
        Returns:
            Self for chaining
        """
        self.conditional_edges[from_node] = {
            "condition": condition,
            "routes": routes
        }
        
        # Convert END string to actual END constant in routes
        processed_routes = {}
        for key, value in routes.items():
            if value == "END":
                processed_routes[key] = END
            else:
                processed_routes[key] = value
        
        self.state_graph.add_conditional_edges(
            from_node,
            condition,
            processed_routes
        )
        
        return self
    
    def add_parallel_nodes(
        self,
        nodes: List[str],
        aggregator: Callable,
        aggregator_name: str = "aggregator"
    ) -> "WorkflowBuilder":
        """Add nodes that execute in parallel.
        
        Args:
            nodes: List of node names to execute in parallel
            aggregator: Function to aggregate results
            aggregator_name: Name for aggregator node
            
        Returns:
            Self for chaining
        """
        # Add aggregator node
        self.add_node(aggregator_name, aggregator, "Aggregate parallel results")
        
        # Connect all parallel nodes to aggregator
        for node_name in nodes:
            if node_name in self.nodes:
                self.add_edge(node_name, aggregator_name)
        
        return self
    
    def set_entry_point(self, node_name: str) -> "WorkflowBuilder":
        """Set the entry point for the workflow.
        
        Args:
            node_name: Name of entry node
            
        Returns:
            Self for chaining
        """
        self.entry_point = node_name
        self.state_graph.add_edge(START, node_name)
        return self
    
    def compile(
        self,
        checkpointer: Optional[Any] = None
    ) -> CompiledStateGraph:
        """Compile the workflow into executable graph.
        
        Args:
            checkpointer: Optional checkpointer for memory
            
        Returns:
            Compiled workflow graph
        """
        # Add entry edge if not already added
        if self.entry_point and START not in [e.from_node for e in self.edges]:
            self.state_graph.add_edge(START, self.entry_point)
        
        # Create checkpointer if memory is enabled
        if self.config.enable_memory and not checkpointer:
            if self.config.memory_type == "in_memory":
                checkpointer = MemorySaver()
            elif self.config.memory_type == "persistent":
                from pathlib import Path
                db_path = Path(f"workflows/{self.config.name}_checkpoints.db")
                db_path.parent.mkdir(parents=True, exist_ok=True)
                checkpointer = SqliteSaver(conn_string=str(db_path))
        
        # Compile the graph
        compiled = self.state_graph.compile(
            checkpointer=checkpointer,
            debug=self.config.debug
        )
        
        # Visualize if requested
        if self.config.visualize:
            self._visualize_graph(compiled)
        
        return compiled
    
    def _wrap_with_retry(
        self,
        func: Callable,
        max_retries: int
    ) -> Callable:
        """Wrap a function with retry logic.
        
        Args:
            func: Function to wrap
            max_retries: Maximum retry attempts
            
        Returns:
            Wrapped function
        """
        def wrapped(state: WorkflowState) -> WorkflowState:
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    return func(state)
                except Exception as e:
                    last_error = str(e)
                    retries += 1
                    if retries >= max_retries:
                        state["error"] = f"Failed after {max_retries} retries: {last_error}"
                        break
            
            return state
        
        return wrapped
    
    def _visualize_graph(self, compiled: CompiledStateGraph):
        """Visualize the compiled graph.
        
        Args:
            compiled: Compiled graph
        """
        try:
            # Try to generate visualization
            png_data = compiled.get_graph().draw_mermaid_png()
            
            # Save to file
            from pathlib import Path
            output_dir = Path("visualizations")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{self.config.name}_workflow.png"
            with open(output_path, "wb") as f:
                f.write(png_data)
            
            print(f"📊 Workflow visualization saved to {output_path}")
            
            # Also print ASCII representation
            print("\n📈 Workflow structure:")
            print(compiled.get_graph().draw_ascii())
            
        except Exception as e:
            print(f"⚠️ Could not visualize workflow: {e}")
    
    def validate(self) -> List[str]:
        """Validate the workflow configuration.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Check for orphaned nodes
        connected_nodes = set()
        for edge in self.edges:
            if edge.from_node != "START":
                connected_nodes.add(edge.from_node)
            if edge.to_node != "END":
                connected_nodes.add(edge.to_node)
        
        for node_name in self.conditional_edges.keys():
            connected_nodes.add(node_name)
        
        orphaned = set(self.nodes.keys()) - connected_nodes
        if orphaned and self.entry_point not in orphaned:
            issues.append(f"Orphaned nodes detected: {orphaned}")
        
        # Check for circular dependencies (simplified)
        if len(self.nodes) > self.config.max_steps:
            issues.append(f"Number of nodes ({len(self.nodes)}) might exceed max steps ({self.config.max_steps})")
        
        # Check entry point
        if not self.entry_point:
            issues.append("No entry point defined")
        
        return issues