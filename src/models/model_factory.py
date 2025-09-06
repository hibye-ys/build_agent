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
    GOOGLE = "google"  # Changed from google_genai to google


@dataclass
class ModelConfig:
    """Configuration for model initialization."""

    provider: ModelProvider
    model_name: str
    temperature: float = 0.5
    max_tokens: Optional[int] = None
    streaming: bool = False
    additional_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_kwargs is None:
            self.additional_kwargs = {}

        # Convert string provider to ModelProvider enum
        if isinstance(self.provider, str):
            try:
                self.provider = ModelProvider(self.provider)
            except ValueError:
                raise ValueError(
                    f"Unsupported provider: {self.provider}. Supported: {list(ModelProvider)}"
                )

        # Convert string temperature to float
        if isinstance(self.temperature, str):
            self.temperature = float(self.temperature)


# Model registry with available models per provider (Updated September 2025)
MODEL_REGISTRY = {
    ModelProvider.OPENAI: {
        # GPT-5 Series (Latest flagship - September 2025)
        "gpt-5": "gpt-5",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
        # GPT-4.1 Series (Specialized for coding - June 2025)
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-nano": "gpt-4.1-nano",
        # O-Series (Reasoning models)
        "o3": "o3",
        "o4-mini": "o4-mini",
    },
    ModelProvider.ANTHROPIC: {
        # Claude 4 Series (Latest - August 2025)
        "claude-opus-4.1": "claude-opus-4.1",
        "claude-sonnet-4": "claude-sonnet-4",
        # Claude 3.7 Series (February 2025)
        "claude-3.7-sonnet": "claude-3.7-sonnet",
    },
    ModelProvider.GOOGLE: {
        # Gemini 2.5 Series (Latest - September 2025)
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    },
}


def get_available_models(
    provider: Optional[ModelProvider] = None,
) -> Dict[str, List[str]]:
    """Get available models for all or specific provider.

    Args:
        provider: Optional specific provider to query

    Returns:
        Dictionary mapping provider to list of available model names
    """
    if provider:
        return {provider.value: list(MODEL_REGISTRY.get(provider, {}).keys())}

    return {p.value: list(models.keys()) for p, models in MODEL_REGISTRY.items()}


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
        ModelProvider.GOOGLE: "GOOGLE_API_KEY",
    }

    api_key_env = api_key_mapping.get(config.provider)
    if api_key_env and not os.getenv(api_key_env):
        raise ValueError(f"Missing {api_key_env} environment variable")

    # Get full model name
    model_registry = MODEL_REGISTRY.get(config.provider, {})
    full_model_name = model_registry.get(config.model_name, config.model_name)

    # Build base kwargs (common to all providers)
    base_kwargs = {"temperature": config.temperature, **config.additional_kwargs}

    if config.max_tokens:
        base_kwargs["max_tokens"] = config.max_tokens

    # Create model using provider-specific class with provider-specific parameters
    if config.provider == ModelProvider.OPENAI:
        # OpenAI supports streaming parameter
        openai_kwargs = {
            **base_kwargs,
            "streaming": config.streaming,  # OpenAI explicitly supports this
        }
        return ChatOpenAI(
            model=full_model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            **openai_kwargs,
        )
    elif config.provider == ModelProvider.ANTHROPIC:
        # Anthropic supports streaming parameter
        anthropic_kwargs = {
            **base_kwargs,
            "streaming": config.streaming,  # Anthropic explicitly supports this
        }
        return ChatAnthropic(
            model=full_model_name,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            **anthropic_kwargs,
        )
    elif config.provider == ModelProvider.GOOGLE:
        # Google models may not support streaming in the same way
        # Only pass streaming if explicitly requested and not default
        google_kwargs = base_kwargs.copy()
        if (
            config.streaming and config.streaming != False
        ):  # Only if explicitly set to True
            google_kwargs["streaming"] = config.streaming
        return ChatGoogleGenerativeAI(
            model=full_model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            **google_kwargs,
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
    config = ModelConfig(provider=provider, model_name=model_name, **kwargs)

    return create_chat_model(config)
