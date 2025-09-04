"""Model factory for creating chat models from different providers."""

import os
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

# Load environment variables
load_dotenv()


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google_genai"


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = False
    additional_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_kwargs is None:
            self.additional_kwargs = {}


# Model registry with available models per provider
MODEL_REGISTRY = {
    ModelProvider.OPENAI: {
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-turbo-preview",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
    },
    ModelProvider.ANTHROPIC: {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku": "claude-3-5-haiku-latest",
    },
    ModelProvider.GOOGLE: {
        "gemini-pro": "gemini-pro",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-2.0-flash": "gemini-2.0-flash-exp",
    }
}


def get_available_models(provider: Optional[ModelProvider] = None) -> Dict[str, List[str]]:
    """Get available models for all or specific provider.
    
    Args:
        provider: Optional specific provider to query
        
    Returns:
        Dictionary mapping provider to list of available model names
    """
    if provider:
        return {provider.value: list(MODEL_REGISTRY.get(provider, {}).keys())}
    
    return {
        p.value: list(models.keys())
        for p, models in MODEL_REGISTRY.items()
    }


def create_chat_model(config: ModelConfig) -> BaseChatModel:
    """Create a chat model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized chat model
        
    Raises:
        ValueError: If provider is not supported or API key is missing
    """
    # Validate API keys
    api_key_mapping = {
        ModelProvider.OPENAI: "OPENAI_API_KEY",
        ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        ModelProvider.GOOGLE: "GOOGLE_API_KEY"
    }
    
    api_key_env = api_key_mapping.get(config.provider)
    if api_key_env and not os.getenv(api_key_env):
        raise ValueError(f"Missing {api_key_env} environment variable")
    
    # Get full model name
    model_registry = MODEL_REGISTRY.get(config.provider, {})
    full_model_name = model_registry.get(config.model_name, config.model_name)
    
    # Build common kwargs
    kwargs = {
        "temperature": config.temperature,
        "streaming": config.streaming,
        **config.additional_kwargs
    }
    
    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens
    
    # Create model using provider-specific class
    if config.provider == ModelProvider.OPENAI:
        return ChatOpenAI(
            model=full_model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs
        )
    elif config.provider == ModelProvider.ANTHROPIC:
        return ChatAnthropic(
            model=full_model_name,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            **kwargs
        )
    elif config.provider == ModelProvider.GOOGLE:
        return ChatGoogleGenerativeAI(
            model=full_model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


def create_chat_model_from_string(model_string: str, **kwargs) -> BaseChatModel:
    """Create a chat model from a string specification.
    
    Args:
        model_string: String in format "provider:model_name" or just "model_name"
        **kwargs: Additional model configuration
        
    Returns:
        Initialized chat model
        
    Examples:
        >>> model = create_chat_model_from_string("openai:gpt-4")
        >>> model = create_chat_model_from_string("anthropic:claude-3-opus")
        >>> model = create_chat_model_from_string("gpt-4")  # Uses default provider
    """
    # Parse model string
    if ":" in model_string:
        provider_str, model_name = model_string.split(":", 1)
        provider = ModelProvider(provider_str)
    else:
        # Use default provider from environment or OpenAI
        default_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
        provider = ModelProvider(default_provider)
        model_name = model_string
    
    # Create config
    config = ModelConfig(
        provider=provider,
        model_name=model_name,
        **kwargs
    )
    
    return create_chat_model(config)