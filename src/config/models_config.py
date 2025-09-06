"""Model configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..models.model_factory import ModelProvider, get_available_models, MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelConfigManager:
    """Manages model configuration from YAML files."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the model configuration manager.
        
        Args:
            config_file: Path to the models config YAML file.
                        If None, uses default config/models_config.yaml
        """
        if config_file is None:
            # Default to config/models_config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / "config" / "models_config.yaml"
        
        self.config_file = Path(config_file)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if not self.config_file.exists():
                logger.warning(f"Config file not found: {self.config_file}")
                return self._get_default_config()
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate config structure
            self._validate_config(config)
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_file}: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and model availability."""
        required_sections = ['default_models', 'available_models', 'examples']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate that default models are available
        for provider, model in config['default_models'].items():
            if not self.validate_model_availability(provider, model):
                logger.warning(f"Default model {model} not available for provider {provider}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            'default_models': {
                'openai': 'gpt-4.1-mini',
                'anthropic': 'claude-sonnet-4',
                'google': 'gemini-2.5-flash'
            },
            'available_models': get_available_models(),
            'examples': {},
            'temperature_settings': {
                'creative': 0.8,
                'analytical': 0.3,
                'balanced': 0.5,
                'deterministic': 0.0
            }
        }
    
    def get_default_model(self, provider: str) -> Optional[str]:
        """Get default model for a provider.
        
        Args:
            provider: Provider name (openai, anthropic, google)
            
        Returns:
            Default model name or None if provider not found
        """
        return self._config.get('default_models', {}).get(provider.lower())
    
    def get_available_models_for_provider(self, provider: str) -> List[str]:
        """Get list of available models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of available model names
        """
        # First check config file
        config_models = self._config.get('available_models', {}).get(provider.lower(), [])
        if config_models:
            return config_models
        
        # Fallback to MODEL_REGISTRY
        try:
            provider_enum = ModelProvider(provider.lower())
            registry_models = list(MODEL_REGISTRY.get(provider_enum, {}).keys())
            return registry_models
        except ValueError:
            logger.warning(f"Unknown provider: {provider}")
            return []
    
    def validate_model_availability(self, provider: str, model: str) -> bool:
        """Validate if a model is available for the given provider.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            True if model is available, False otherwise
        """
        available_models = self.get_available_models_for_provider(provider)
        return model in available_models
    
    def get_model_config_for_example(self, example_name: str) -> Dict[str, Any]:
        """Get model configuration for a specific example.
        
        Args:
            example_name: Name of the example
            
        Returns:
            Dictionary with model configurations for the example
        """
        example_config = self._config.get('examples', {}).get(example_name, {})
        
        # If no specific config, use defaults
        if not example_config:
            return {
                'openai_model': self.get_default_model('openai'),
                'anthropic_model': self.get_default_model('anthropic'),
                'google_model': self.get_default_model('google')
            }
        
        return example_config
    
    def get_temperature_for_use_case(self, use_case: str) -> float:
        """Get temperature setting for a use case.
        
        Args:
            use_case: Use case name (creative, analytical, balanced, deterministic)
            
        Returns:
            Temperature value
        """
        return self._config.get('temperature_settings', {}).get(use_case, 0.5)
    
    def suggest_alternative_model(self, provider: str, unavailable_model: str) -> Optional[str]:
        """Suggest an alternative model if the requested one is not available.
        
        Args:
            provider: Provider name
            unavailable_model: The model that's not available
            
        Returns:
            Alternative model name or None
        """
        available_models = self.get_available_models_for_provider(provider)
        if not available_models:
            return None
        
        # Try to find a similar model (e.g., if gpt-4 not available, suggest gpt-4.1)
        if 'gpt-4' in unavailable_model and any('gpt-4' in model for model in available_models):
            gpt4_models = [m for m in available_models if 'gpt-4' in m]
            return gpt4_models[0]
        
        if 'claude' in unavailable_model and any('claude' in model for model in available_models):
            claude_models = [m for m in available_models if 'claude' in m]
            return claude_models[0]
        
        if 'gemini' in unavailable_model and any('gemini' in model for model in available_models):
            gemini_models = [m for m in available_models if 'gemini' in m]
            return gemini_models[0]
        
        # Otherwise return default model for provider
        return self.get_default_model(provider)


# Global instance
_config_manager = None


def load_models_config(config_file: Optional[str] = None) -> ModelConfigManager:
    """Load the models configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        ModelConfigManager instance
    """
    global _config_manager
    if _config_manager is None or config_file is not None:
        _config_manager = ModelConfigManager(config_file)
    return _config_manager


def get_default_model(provider: str) -> Optional[str]:
    """Get default model for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Default model name
    """
    manager = load_models_config()
    return manager.get_default_model(provider)


def get_available_models_for_provider(provider: str) -> List[str]:
    """Get available models for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        List of available model names
    """
    manager = load_models_config()
    return manager.get_available_models_for_provider(provider)


def validate_model_availability(provider: str, model: str) -> bool:
    """Validate model availability.
    
    Args:
        provider: Provider name  
        model: Model name
        
    Returns:
        True if available
    """
    manager = load_models_config()
    return manager.validate_model_availability(provider, model)


def get_model_config_for_example(example_name: str) -> Dict[str, Any]:
    """Get model config for an example.
    
    Args:
        example_name: Example name
        
    Returns:
        Model configuration dictionary
    """
    manager = load_models_config()
    return manager.get_model_config_for_example(example_name)


def get_validated_model_config(provider: str, requested_model: Optional[str] = None) -> Tuple[str, bool]:
    """Get a validated model configuration.
    
    Args:
        provider: Provider name
        requested_model: Requested model name (optional)
        
    Returns:
        Tuple of (model_name, is_original_request)
        is_original_request is False if we had to use an alternative
    """
    manager = load_models_config()
    
    # If no model requested, use default
    if not requested_model:
        default_model = manager.get_default_model(provider)
        if default_model and manager.validate_model_availability(provider, default_model):
            return default_model, True
        else:
            # Even default is not available, get any available model
            available_models = manager.get_available_models_for_provider(provider)
            if available_models:
                logger.warning(f"Default model not available for {provider}, using {available_models[0]}")
                return available_models[0], False
            else:
                raise ValueError(f"No models available for provider {provider}")
    
    # Check if requested model is available
    if manager.validate_model_availability(provider, requested_model):
        return requested_model, True
    
    # Try to find alternative
    alternative = manager.suggest_alternative_model(provider, requested_model)
    if alternative:
        logger.warning(f"Model {requested_model} not available for {provider}, using {alternative}")
        return alternative, False
    
    raise ValueError(f"Model {requested_model} not available for provider {provider} and no alternatives found")