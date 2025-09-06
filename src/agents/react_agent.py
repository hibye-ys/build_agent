"""ReAct agent implementation using LangGraph."""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from ..models import ModelConfig, create_chat_model
from ..tools import get_tools_by_names


@dataclass
class ReactAgentConfig:
    """Configuration for ReAct agent."""
    model_config: ModelConfig
    tools: List[Union[str, BaseTool]]
    system_prompt: Optional[str] = None
    memory: bool = False
    streaming: bool = False
    verbose: bool = False
    max_iterations: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")


def create_react_agent_wrapper(
    config: ReactAgentConfig
) -> CompiledStateGraph:
    """Create a ReAct agent with specified configuration.
    
    Args:
        config: Agent configuration
        
    Returns:
        Compiled agent graph
    """
    # Create model
    model = create_chat_model(config.model_config)
    
    # Process tools
    tools = []
    for tool_spec in config.tools:
        if isinstance(tool_spec, str):
            # Get tool by name from registry
            tool_list = get_tools_by_names(tool_spec)
            tools.extend(tool_list)
        elif isinstance(tool_spec, BaseTool):
            tools.append(tool_spec)
        else:
            raise ValueError(f"Invalid tool specification: {tool_spec}")
    
    if not tools:
        raise ValueError("At least one tool must be provided")
    
    # Create prompt if provided
    prompt = None
    if config.system_prompt:
        prompt = config.system_prompt
    
    # Create checkpointer if memory is enabled
    checkpointer = None
    if config.memory:
        checkpointer = MemorySaver()
    
    # Create the ReAct agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer,
        debug=config.verbose
    )
    
    return agent


def create_simple_react_agent(
    model_provider: str = "openai",
    model_name: str = "gpt-4",
    tool_names: List[str] = None,
    system_prompt: str = None,
    memory: bool = False
) -> CompiledStateGraph:
    """Create a simple ReAct agent with minimal configuration.
    
    Args:
        model_provider: Model provider name
        model_name: Model name
        tool_names: List of tool names to use
        system_prompt: Optional system prompt
        memory: Whether to enable memory
        
    Returns:
        Compiled agent graph
        
    Example:
        >>> agent = create_simple_react_agent(
        ...     model_provider="openai",
        ...     model_name="gpt-4",
        ...     tool_names=["calculator", "web_search"],
        ...     system_prompt="You are a helpful assistant."
        ... )
    """
    if tool_names is None:
        tool_names = ["calculator", "weather", "datetime"]
    
    from ..models import ModelProvider
    
    # Map string to enum
    provider_map = {
        "openai": ModelProvider.OPENAI,
        "anthropic": ModelProvider.ANTHROPIC,
        "google": ModelProvider.GOOGLE,
        "google_genai": ModelProvider.GOOGLE
    }
    
    provider_enum = provider_map.get(model_provider.lower())
    if not provider_enum:
        raise ValueError(f"Unknown provider: {model_provider}")
    
    config = ReactAgentConfig(
        model_config=ModelConfig(
            provider=provider_enum,
            model_name=model_name
        ),
        tools=tool_names,
        system_prompt=system_prompt,
        memory=memory
    )
    
    return create_react_agent_wrapper(config)