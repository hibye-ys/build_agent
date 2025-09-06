"""Core prompt management system with versioning and A/B testing support."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib
from enum import Enum
import logging

from jinja2 import Template, Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


class PromptEnvironment(Enum):
    """Environment types for prompt selection."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class PromptVersion:
    """Represents a version of a prompt template."""
    version: str
    content: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None
    
    def __post_init__(self):
        """Calculate hash if not provided."""
        if not self.hash:
            self.hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "hash": self.hash
        }


@dataclass
class PromptMetrics:
    """Metrics for prompt performance tracking."""
    prompt_id: str
    version: str
    usage_count: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    avg_tokens_used: int = 0
    feedback_scores: List[float] = field(default_factory=list)
    
    def add_usage(self, success: bool, response_time: float, tokens: int, feedback: Optional[float] = None):
        """Record a usage of this prompt."""
        self.usage_count += 1
        
        # Update success rate
        prev_successes = self.success_rate * (self.usage_count - 1)
        self.success_rate = (prev_successes + (1 if success else 0)) / self.usage_count
        
        # Update average response time
        prev_total_time = self.avg_response_time * (self.usage_count - 1)
        self.avg_response_time = (prev_total_time + response_time) / self.usage_count
        
        # Update average tokens
        prev_total_tokens = self.avg_tokens_used * (self.usage_count - 1)
        self.avg_tokens_used = int((prev_total_tokens + tokens) / self.usage_count)
        
        # Add feedback if provided
        if feedback is not None:
            self.feedback_scores.append(feedback)
    
    @property
    def avg_feedback(self) -> Optional[float]:
        """Calculate average feedback score."""
        if not self.feedback_scores:
            return None
        return sum(self.feedback_scores) / len(self.feedback_scores)


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata and versioning."""
    id: str
    name: str
    description: str
    category: str
    versions: Dict[str, PromptVersion] = field(default_factory=dict)
    current_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, PromptMetrics] = field(default_factory=dict)
    
    def add_version(self, version: str, content: str, set_current: bool = True) -> PromptVersion:
        """Add a new version of the prompt."""
        prompt_version = PromptVersion(
            version=version,
            content=content,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.versions[version] = prompt_version
        
        if set_current or not self.current_version:
            self.current_version = version
        
        # Initialize metrics for this version
        self.metrics[version] = PromptMetrics(
            prompt_id=self.id,
            version=version
        )
        
        return prompt_version
    
    def get_version(self, version: Optional[str] = None) -> Optional[PromptVersion]:
        """Get a specific version or the current version."""
        if version is None:
            version = self.current_version
        
        if version is None:
            return None
        
        return self.versions.get(version)
    
    def render(self, version: Optional[str] = None, **kwargs) -> str:
        """Render the prompt template with provided variables."""
        prompt_version = self.get_version(version)
        if not prompt_version:
            raise ValueError(f"No version found for prompt {self.id}")
        
        template = Template(prompt_version.content)
        return template.render(**kwargs)
    
    def get_metrics(self, version: Optional[str] = None) -> Optional[PromptMetrics]:
        """Get metrics for a specific version."""
        if version is None:
            version = self.current_version
        
        if version is None:
            return None
        
        return self.metrics.get(version)


class PromptTemplateManager:
    """Manages prompt templates with versioning, caching, and A/B testing."""
    
    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        environment: PromptEnvironment = PromptEnvironment.DEVELOPMENT,
        cache_enabled: bool = True,
        default_language: str = "en"
    ):
        """Initialize the prompt template manager.
        
        Args:
            templates_dir: Directory containing template files
            environment: Current environment (dev/staging/prod)
            cache_enabled: Whether to cache rendered templates
            default_language: Default language for prompts
        """
        self.templates_dir = templates_dir or Path("prompts/templates")
        self.environment = environment
        self.cache_enabled = cache_enabled
        self.default_language = default_language
        
        self.templates: Dict[str, PromptTemplate] = {}
        self.cache: Dict[str, str] = {}
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Jinja2 environment
        if self.templates_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            self.jinja_env = Environment(autoescape=False)
    
    def register_template(
        self,
        id: str,
        name: str,
        content: str,
        description: str = "",
        category: str = "general",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        variables: Optional[List[str]] = None
    ) -> PromptTemplate:
        """Register a new prompt template.
        
        Args:
            id: Unique identifier for the template
            name: Human-readable name
            content: Template content with Jinja2 syntax
            description: Description of the template
            category: Category for organization
            version: Initial version
            tags: Tags for search and filtering
            variables: List of variables used in the template
            
        Returns:
            Created PromptTemplate instance
        """
        if id in self.templates:
            logger.warning(f"Template {id} already exists, updating...")
        
        template = PromptTemplate(
            id=id,
            name=name,
            description=description,
            category=category,
            tags=tags or [],
            variables=variables or []
        )
        
        template.add_version(version, content)
        self.templates[id] = template
        
        logger.info(f"Registered template {id} version {version}")
        return template
    
    def get_template(
        self,
        id: str,
        version: Optional[str] = None,
        language: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """Get a prompt template by ID.
        
        Args:
            id: Template ID
            version: Specific version (uses current if not specified)
            language: Language override
            
        Returns:
            PromptTemplate instance or None
        """
        # Check for language-specific template
        if language and language != self.default_language:
            localized_id = f"{id}_{language}"
            if localized_id in self.templates:
                return self.templates[localized_id]
        
        return self.templates.get(id)
    
    def render(
        self,
        id: str,
        version: Optional[str] = None,
        language: Optional[str] = None,
        use_cache: Optional[bool] = None,
        record_metrics: bool = True,
        **variables
    ) -> str:
        """Render a prompt template with variables.
        
        Args:
            id: Template ID
            version: Specific version to render
            language: Language override
            use_cache: Whether to use cache (overrides default)
            record_metrics: Whether to record usage metrics
            **variables: Variables to inject into the template
            
        Returns:
            Rendered prompt string
        """
        template = self.get_template(id, version, language)
        if not template:
            raise ValueError(f"Template {id} not found")
        
        # Check A/B test
        version = self._get_ab_test_version(id) or version
        
        # Check cache
        cache_key = self._get_cache_key(id, version, language, variables)
        if (use_cache if use_cache is not None else self.cache_enabled) and cache_key in self.cache:
            logger.debug(f"Using cached prompt for {id}")
            return self.cache[cache_key]
        
        # Render template
        start_time = datetime.now()
        rendered = template.render(version=version, **variables)
        render_time = (datetime.now() - start_time).total_seconds()
        
        # Cache result
        if self.cache_enabled:
            self.cache[cache_key] = rendered
        
        # Record metrics
        if record_metrics:
            metrics = template.get_metrics(version)
            if metrics:
                # Simple metrics recording (would need actual token counting in production)
                metrics.add_usage(
                    success=True,
                    response_time=render_time,
                    tokens=len(rendered.split())  # Simplified token count
                )
        
        return rendered
    
    def add_version(
        self,
        id: str,
        version: str,
        content: str,
        set_current: bool = False
    ) -> PromptVersion:
        """Add a new version to an existing template.
        
        Args:
            id: Template ID
            version: Version identifier
            content: New content
            set_current: Whether to set as current version
            
        Returns:
            Created PromptVersion instance
        """
        template = self.templates.get(id)
        if not template:
            raise ValueError(f"Template {id} not found")
        
        prompt_version = template.add_version(version, content, set_current)
        
        # Clear cache for this template
        self._clear_template_cache(id)
        
        logger.info(f"Added version {version} to template {id}")
        return prompt_version
    
    def rollback(self, id: str, version: str) -> bool:
        """Rollback to a previous version of a template.
        
        Args:
            id: Template ID
            version: Version to rollback to
            
        Returns:
            True if successful
        """
        template = self.templates.get(id)
        if not template:
            raise ValueError(f"Template {id} not found")
        
        if version not in template.versions:
            raise ValueError(f"Version {version} not found for template {id}")
        
        template.current_version = version
        self._clear_template_cache(id)
        
        logger.info(f"Rolled back template {id} to version {version}")
        return True
    
    def create_ab_test(
        self,
        id: str,
        test_name: str,
        versions: List[str],
        weights: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an A/B test for a template.
        
        Args:
            id: Template ID
            test_name: Name of the test
            versions: List of versions to test
            weights: Weights for each version (equal if not specified)
            metadata: Additional test metadata
            
        Returns:
            Test configuration
        """
        template = self.templates.get(id)
        if not template:
            raise ValueError(f"Template {id} not found")
        
        # Validate versions exist
        for version in versions:
            if version not in template.versions:
                raise ValueError(f"Version {version} not found for template {id}")
        
        # Set equal weights if not provided
        if weights is None:
            weights = [1.0 / len(versions)] * len(versions)
        
        if len(weights) != len(versions):
            raise ValueError("Number of weights must match number of versions")
        
        test_config = {
            "test_name": test_name,
            "template_id": id,
            "versions": versions,
            "weights": weights,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "selection_count": {v: 0 for v in versions}
        }
        
        self.ab_tests[test_name] = test_config
        
        logger.info(f"Created A/B test {test_name} for template {id}")
        return test_config
    
    def _get_ab_test_version(self, template_id: str) -> Optional[str]:
        """Get version for template based on active A/B tests.
        
        Args:
            template_id: Template ID
            
        Returns:
            Selected version or None
        """
        import random
        
        for test_name, test_config in self.ab_tests.items():
            if test_config["template_id"] == template_id:
                # Select version based on weights
                versions = test_config["versions"]
                weights = test_config["weights"]
                selected = random.choices(versions, weights=weights)[0]
                
                # Track selection
                test_config["selection_count"][selected] += 1
                
                return selected
        
        return None
    
    def _get_cache_key(
        self,
        id: str,
        version: Optional[str],
        language: Optional[str],
        variables: Dict[str, Any]
    ) -> str:
        """Generate cache key for a rendered prompt.
        
        Args:
            id: Template ID
            version: Version
            language: Language
            variables: Template variables
            
        Returns:
            Cache key string
        """
        key_parts = [id, version or "current", language or self.default_language]
        
        # Add sorted variables to key
        for k, v in sorted(variables.items()):
            key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(map(str, key_parts))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _clear_template_cache(self, template_id: str):
        """Clear cache entries for a specific template.
        
        Args:
            template_id: Template ID
        """
        keys_to_remove = [
            key for key in self.cache
            if key.startswith(template_id)
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        logger.debug(f"Cleared {len(keys_to_remove)} cache entries for template {template_id}")
    
    def export_metrics(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export metrics for all templates.
        
        Args:
            output_path: Optional path to save metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics_data = {}
        
        for template_id, template in self.templates.items():
            template_metrics = {}
            
            for version, metrics in template.metrics.items():
                template_metrics[version] = {
                    "usage_count": metrics.usage_count,
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "avg_tokens_used": metrics.avg_tokens_used,
                    "avg_feedback": metrics.avg_feedback
                }
            
            metrics_data[template_id] = {
                "name": template.name,
                "current_version": template.current_version,
                "versions": template_metrics
            }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        
        return metrics_data
    
    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[PromptTemplate]:
        """List available templates with optional filtering.
        
        Args:
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            List of matching templates
        """
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if tags:
            templates = [
                t for t in templates
                if any(tag in t.tags for tag in tags)
            ]
        
        return templates