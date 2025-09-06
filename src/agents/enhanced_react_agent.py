"""Enhanced ReAct agent implementation using the new prompt management system."""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from pathlib import Path

from ..models import ModelConfig, create_chat_model
from ..tools import get_tools_by_names
from ..prompts import (
    PromptTemplateManager,
    PromptValidator,
    TokenCounter
)


@dataclass
class EnhancedReactAgentConfig:
    """Enhanced configuration for ReAct agent with prompt management."""
    model_config: ModelConfig
    tools: List[Union[str, BaseTool]]
    prompt_id: Optional[str] = "react_agent"  # Use template ID instead of raw prompt
    prompt_version: Optional[str] = None  # Specific version to use
    prompt_variables: Dict[str, Any] = None  # Variables for prompt rendering
    language: str = "en"  # Language for prompts
    memory: bool = False
    streaming: bool = False
    verbose: bool = False
    max_iterations: int = 10
    use_prompt_optimization: bool = True  # Whether to optimize prompts
    validate_prompt: bool = True  # Whether to validate prompts
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.prompt_variables is None:
            self.prompt_variables = {}


class PromptManagedReactAgent:
    """ReAct agent with integrated prompt management system."""
    
    def __init__(
        self,
        config: EnhancedReactAgentConfig,
        prompt_manager: Optional[PromptTemplateManager] = None,
        environment: str = "development"
    ):
        """Initialize the prompt-managed ReAct agent.
        
        Args:
            config: Agent configuration
            prompt_manager: Optional prompt manager (creates default if not provided)
            environment: Environment for prompt selection
        """
        self.config = config
        self.environment = environment
        
        # Initialize prompt manager if not provided
        if prompt_manager is None:
            self.prompt_manager = self._create_default_prompt_manager()
        else:
            self.prompt_manager = prompt_manager
        
        # Initialize validators
        self.prompt_validator = PromptValidator()
        self.token_counter = TokenCounter(model=config.model_config.model_name)
        
        # Create the agent
        self.agent = self._create_agent()
    
    def _create_default_prompt_manager(self) -> PromptTemplateManager:
        """Create a default prompt manager with templates.
        
        Returns:
            Configured PromptTemplateManager
        """
        # Create prompt manager
        templates_dir = Path("prompts/templates")
        manager = PromptTemplateManager(
            templates_dir=templates_dir,
            environment=self.environment,
            cache_enabled=True,
            default_language=self.config.language
        )
        
        # Load templates from files
        # PromptLoaderManager not available in current implementation
        # loader_manager = PromptLoaderManager(manager)
        
        # Try to load from template files
        # base_file = templates_dir / "base.yaml"
        # if base_file.exists():
        #     loader_manager.load_from_file(base_file)
        
        # Load language-specific templates
        # if self.config.language != "en":
        #     lang_file = templates_dir / "i18n" / f"{self.config.language}.yaml"
        #     if lang_file.exists():
        #         loader_manager.load_from_file(lang_file)
        
        return manager
    
    def _create_agent(self) -> CompiledStateGraph:
        """Create the ReAct agent with prompt management.
        
        Returns:
            Compiled agent graph
        """
        # Create model
        model = create_chat_model(self.config.model_config)
        
        # Process tools
        tools = self._process_tools()
        
        # Get and render prompt
        prompt = self._get_rendered_prompt(tools)
        
        # Validate prompt if enabled
        if self.config.validate_prompt:
            self._validate_prompt(prompt)
        
        # Optimize prompt if enabled
        if self.config.use_prompt_optimization:
            prompt = self._optimize_prompt(prompt)
        
        # Create checkpointer if memory is enabled
        checkpointer = None
        if self.config.memory:
            checkpointer = MemorySaver()
        
        # Create the ReAct agent
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=prompt,
            checkpointer=checkpointer,
            debug=self.config.verbose
        )
        
        return agent
    
    def _process_tools(self) -> List[BaseTool]:
        """Process and validate tools.
        
        Returns:
            List of tool instances
        """
        tools = []
        for tool_spec in self.config.tools:
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
        
        return tools
    
    def _get_rendered_prompt(self, tools: List[BaseTool]) -> str:
        """Get and render the prompt template.
        
        Args:
            tools: List of available tools
            
        Returns:
            Rendered prompt string
        """
        # Prepare variables
        variables = self.config.prompt_variables.copy()
        
        # Add default variables
        variables.setdefault("agent_name", "AI Assistant")
        variables.setdefault("task", "Help the user with their request")
        
        # Add tools information
        if tools:
            variables["tools"] = [
                f"{tool.name}: {tool.description}" for tool in tools
            ]
        
        # Try language-specific prompt first
        prompt_id = self.config.prompt_id
        if self.config.language != "en":
            localized_id = f"{prompt_id}_{self.config.language}"
            if self.prompt_manager.get_template(localized_id):
                prompt_id = localized_id
        
        # Render the prompt
        try:
            rendered = self.prompt_manager.render(
                id=prompt_id,
                version=self.config.prompt_version,
                language=self.config.language,
                record_metrics=True,
                **variables
            )
            return rendered
        except ValueError as e:
            # Fallback to a simple default prompt
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}" for tool in tools
            ])
            return f"""You are a helpful AI assistant with access to tools.

Available tools:
{tool_descriptions}

Think step by step to solve the user's request."""
    
    def _validate_prompt(self, prompt: str):
        """Validate the prompt for issues.
        
        Args:
            prompt: Prompt to validate
        """
        result = self.prompt_validator.validate(
            prompt,
            variables=self.config.prompt_variables
        )
        
        # Log any issues
        if result.errors:
            for error in result.errors:
                print(f"❌ Prompt Error: {error}")
        
        if result.warnings:
            for warning in result.warnings:
                print(f"⚠️ Prompt Warning: {warning}")
        
        if result.suggestions:
            for suggestion in result.suggestions:
                print(f"💡 Prompt Suggestion: {suggestion}")
        
        # Show token count
        if "token_count" in result.metrics:
            print(f"📊 Token count: {result.metrics['token_count']}")
    
    def _optimize_prompt(self, prompt: str) -> str:
        """Optimize the prompt for token efficiency.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Optimized prompt
        """
        from ..prompts import PromptOptimizer
        
        optimizer = PromptOptimizer(target_reduction=0.2)
        optimized, metrics = optimizer.optimize(prompt, preserve_meaning=True)
        
        if metrics["target_met"]:
            print(f"✅ Prompt optimized: {metrics['original_tokens']} → {metrics['optimized_tokens']} tokens")
            return optimized
        else:
            return prompt
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with input.
        
        Args:
            input_data: Input dictionary
            
        Returns:
            Agent response
        """
        return self.agent.invoke(input_data)
    
    def stream(self, input_data: Dict[str, Any]):
        """Stream agent responses.
        
        Args:
            input_data: Input dictionary
            
        Yields:
            Response chunks
        """
        return self.agent.stream(input_data)
    
    def update_prompt_version(self, version: str):
        """Update to a different prompt version.
        
        Args:
            version: Version to switch to
        """
        self.config.prompt_version = version
        self.agent = self._create_agent()
    
    def switch_language(self, language: str):
        """Switch to a different language.
        
        Args:
            language: Language code (e.g., 'ko', 'es')
        """
        self.config.language = language
        self.agent = self._create_agent()
    
    def get_prompt_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics for the current prompt.
        
        Returns:
            Metrics dictionary or None
        """
        template = self.prompt_manager.get_template(
            self.config.prompt_id,
            language=self.config.language
        )
        
        if template:
            metrics = template.get_metrics(self.config.prompt_version)
            if metrics:
                return {
                    "usage_count": metrics.usage_count,
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "avg_tokens_used": metrics.avg_tokens_used,
                    "avg_feedback": metrics.avg_feedback
                }
        
        return None


def create_enhanced_react_agent(
    model_provider: str = "openai",
    model_name: str = "gpt-4",
    tool_names: List[str] = None,
    prompt_id: str = "react_agent",
    language: str = "en",
    memory: bool = False,
    environment: str = "development"
) -> PromptManagedReactAgent:
    """Create an enhanced ReAct agent with prompt management.
    
    Args:
        model_provider: Model provider name
        model_name: Model name
        tool_names: List of tool names to use
        prompt_id: Prompt template ID to use
        language: Language for prompts
        memory: Whether to enable memory
        environment: Environment (development/staging/production)
        
    Returns:
        PromptManagedReactAgent instance
        
    Example:
        >>> agent = create_enhanced_react_agent(
        ...     model_provider="openai",
        ...     model_name="gpt-4",
        ...     tool_names=["calculator", "web_search"],
        ...     language="ko",  # Korean prompts
        ...     memory=True
        ... )
        >>> response = agent.invoke({"messages": [HumanMessage("안녕하세요")]})
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
    
    # Map environment string to enum
    # PromptEnvironment enum not available in current implementation
    # Using string representation directly
    env_string = environment.lower() if environment else "development"
    
    config = EnhancedReactAgentConfig(
        model_config=ModelConfig(
            provider=provider_enum,
            model_name=model_name
        ),
        tools=tool_names,
        prompt_id=prompt_id,
        language=language,
        memory=memory
    )
    
    return PromptManagedReactAgent(config, environment=env_string)