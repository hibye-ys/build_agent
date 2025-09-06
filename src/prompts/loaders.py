"""Loaders for importing prompts from various sources."""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod
import json
import yaml
import logging
from datetime import datetime

from .prompt_manager import PromptTemplateManager, PromptTemplate, PromptEnvironment
from .templates import BasePromptTemplate, PromptTemplateFactory

logger = logging.getLogger(__name__)


class PromptLoader(ABC):
    """Abstract base class for prompt loaders."""
    
    @abstractmethod
    def load(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load prompts from source.
        
        Args:
            source: Source to load from
            
        Returns:
            Dictionary of prompt data
        """
        pass
    
    @abstractmethod
    def save(self, data: Dict[str, Any], destination: Union[str, Path]) -> bool:
        """Save prompts to destination.
        
        Args:
            data: Prompt data to save
            destination: Where to save
            
        Returns:
            True if successful
        """
        pass


class YAMLPromptLoader(PromptLoader):
    """Loader for YAML prompt files."""
    
    def load(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load prompts from YAML file.
        
        Args:
            source: Path to YAML file
            
        Returns:
            Dictionary of prompt data
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        
        with open(source_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        logger.info(f"Loaded prompts from {source_path}")
        return data
    
    def save(self, data: Dict[str, Any], destination: Union[str, Path]) -> bool:
        """Save prompts to YAML file.
        
        Args:
            data: Prompt data to save
            destination: Path to save to
            
        Returns:
            True if successful
        """
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved prompts to {dest_path}")
        return True


class JSONPromptLoader(PromptLoader):
    """Loader for JSON prompt files."""
    
    def load(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load prompts from JSON file.
        
        Args:
            source: Path to JSON file
            
        Returns:
            Dictionary of prompt data
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded prompts from {source_path}")
        return data
    
    def save(self, data: Dict[str, Any], destination: Union[str, Path]) -> bool:
        """Save prompts to JSON file.
        
        Args:
            data: Prompt data to save
            destination: Path to save to
            
        Returns:
            True if successful
        """
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved prompts to {dest_path}")
        return True


class FilePromptLoader:
    """Unified loader that handles multiple file formats."""
    
    def __init__(self):
        """Initialize with supported loaders."""
        self.loaders = {
            '.yaml': YAMLPromptLoader(),
            '.yml': YAMLPromptLoader(),
            '.json': JSONPromptLoader()
        }
    
    def load(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load prompts from file based on extension.
        
        Args:
            source: Path to file
            
        Returns:
            Dictionary of prompt data
        """
        source_path = Path(source)
        ext = source_path.suffix.lower()
        
        loader = self.loaders.get(ext)
        if not loader:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return loader.load(source_path)
    
    def save(
        self,
        data: Dict[str, Any],
        destination: Union[str, Path],
        format: Optional[str] = None
    ) -> bool:
        """Save prompts to file.
        
        Args:
            data: Prompt data to save
            destination: Path to save to
            format: Force specific format (yaml/json)
            
        Returns:
            True if successful
        """
        dest_path = Path(destination)
        
        if format:
            ext = f".{format.lower()}"
        else:
            ext = dest_path.suffix.lower()
        
        loader = self.loaders.get(ext)
        if not loader:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return loader.save(data, dest_path)
    
    def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        pattern: str = "*"
    ) -> Dict[str, Dict[str, Any]]:
        """Load all prompt files from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            pattern: File pattern to match
            
        Returns:
            Dictionary mapping file names to prompt data
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        results = {}
        
        # Find matching files
        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)
        
        for file_path in files:
            if file_path.suffix.lower() in self.loaders:
                try:
                    data = self.load(file_path)
                    results[file_path.stem] = data
                    logger.info(f"Loaded {file_path.stem} from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        return results


class PromptLoaderManager:
    """Manages loading prompts into PromptTemplateManager."""
    
    def __init__(self, prompt_manager: PromptTemplateManager):
        """Initialize with a prompt manager.
        
        Args:
            prompt_manager: PromptTemplateManager instance
        """
        self.prompt_manager = prompt_manager
        self.file_loader = FilePromptLoader()
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        environment: Optional[PromptEnvironment] = None
    ) -> int:
        """Load prompts from a file into the manager.
        
        Args:
            file_path: Path to prompt file
            environment: Environment to load for
            
        Returns:
            Number of prompts loaded
        """
        data = self.file_loader.load(file_path)
        return self._process_prompt_data(data, environment)
    
    def load_from_directory(
        self,
        directory: Union[str, Path],
        environment: Optional[PromptEnvironment] = None,
        recursive: bool = True
    ) -> int:
        """Load all prompts from a directory.
        
        Args:
            directory: Directory path
            environment: Environment to load for
            recursive: Whether to search recursively
            
        Returns:
            Total number of prompts loaded
        """
        all_data = self.file_loader.load_directory(directory, recursive)
        
        total_loaded = 0
        for file_name, data in all_data.items():
            count = self._process_prompt_data(data, environment)
            total_loaded += count
            logger.info(f"Loaded {count} prompts from {file_name}")
        
        return total_loaded
    
    def _process_prompt_data(
        self,
        data: Dict[str, Any],
        environment: Optional[PromptEnvironment] = None
    ) -> int:
        """Process loaded prompt data and register with manager.
        
        Args:
            data: Prompt data dictionary
            environment: Environment to load for
            
        Returns:
            Number of prompts registered
        """
        count = 0
        
        # Check for environment-specific prompts
        if environment and "environments" in data:
            env_data = data["environments"].get(environment.value, {})
            if "prompts" in env_data:
                data = env_data
        
        # Process prompts
        prompts = data.get("prompts", [])
        
        for prompt_data in prompts:
            try:
                self._register_prompt(prompt_data)
                count += 1
            except Exception as e:
                logger.error(f"Failed to register prompt: {e}")
        
        return count
    
    def _register_prompt(self, prompt_data: Dict[str, Any]):
        """Register a single prompt with the manager.
        
        Args:
            prompt_data: Prompt configuration
        """
        # Extract basic fields
        prompt_id = prompt_data.get("id")
        name = prompt_data.get("name", prompt_id)
        description = prompt_data.get("description", "")
        category = prompt_data.get("category", "general")
        
        # Handle template content
        template_content = prompt_data.get("template", "")
        
        # Handle versions
        versions = prompt_data.get("versions", {})
        
        if not versions:
            # Single version prompt
            version = prompt_data.get("version", "1.0.0")
            versions = {version: {"content": template_content}}
        
        # Register the template
        first_version = list(versions.keys())[0]
        first_content = versions[first_version].get("content", template_content)
        
        template = self.prompt_manager.register_template(
            id=prompt_id,
            name=name,
            content=first_content,
            description=description,
            category=category,
            version=first_version,
            tags=prompt_data.get("tags", []),
            variables=prompt_data.get("variables", [])
        )
        
        # Add additional versions
        for version, version_data in versions.items():
            if version != first_version:
                content = version_data.get("content", template_content)
                self.prompt_manager.add_version(
                    prompt_id,
                    version,
                    content,
                    set_current=version_data.get("current", False)
                )
    
    def export_to_file(
        self,
        destination: Union[str, Path],
        format: Optional[str] = None,
        include_metrics: bool = False
    ) -> bool:
        """Export all prompts to a file.
        
        Args:
            destination: Where to save
            format: File format (yaml/json)
            include_metrics: Whether to include metrics
            
        Returns:
            True if successful
        """
        data = self._export_prompt_data(include_metrics)
        return self.file_loader.save(data, destination, format)
    
    def _export_prompt_data(self, include_metrics: bool = False) -> Dict[str, Any]:
        """Export prompt data from manager.
        
        Args:
            include_metrics: Whether to include metrics
            
        Returns:
            Dictionary of prompt data
        """
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "prompts": []
        }
        
        for template in self.prompt_manager.templates.values():
            prompt_data = {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "tags": template.tags,
                "variables": template.variables,
                "versions": {}
            }
            
            # Add versions
            for version_id, version in template.versions.items():
                prompt_data["versions"][version_id] = {
                    "content": version.content,
                    "created_at": version.created_at.isoformat(),
                    "updated_at": version.updated_at.isoformat(),
                    "current": version_id == template.current_version
                }
            
            # Add metrics if requested
            if include_metrics:
                prompt_data["metrics"] = {}
                for version_id, metrics in template.metrics.items():
                    prompt_data["metrics"][version_id] = {
                        "usage_count": metrics.usage_count,
                        "success_rate": metrics.success_rate,
                        "avg_response_time": metrics.avg_response_time,
                        "avg_tokens_used": metrics.avg_tokens_used
                    }
            
            data["prompts"].append(prompt_data)
        
        return data


class EnvironmentPromptLoader:
    """Loads prompts based on environment configuration."""
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        environment: PromptEnvironment = PromptEnvironment.DEVELOPMENT
    ):
        """Initialize environment-aware loader.
        
        Args:
            base_dir: Base directory for prompts
            environment: Current environment
        """
        self.base_dir = Path(base_dir)
        self.environment = environment
        self.file_loader = FilePromptLoader()
    
    def load_for_environment(
        self,
        prompt_manager: PromptTemplateManager
    ) -> int:
        """Load prompts appropriate for current environment.
        
        Args:
            prompt_manager: Manager to load into
            
        Returns:
            Number of prompts loaded
        """
        loader_manager = PromptLoaderManager(prompt_manager)
        
        # Load base prompts
        base_file = self.base_dir / "base.yaml"
        count = 0
        
        if base_file.exists():
            count += loader_manager.load_from_file(base_file)
        
        # Load environment-specific prompts
        env_file = self.base_dir / f"{self.environment.value}.yaml"
        
        if env_file.exists():
            count += loader_manager.load_from_file(env_file, self.environment)
        
        # Load from environment directory
        env_dir = self.base_dir / self.environment.value
        
        if env_dir.exists():
            count += loader_manager.load_from_directory(env_dir, self.environment)
        
        logger.info(f"Loaded {count} prompts for {self.environment.value} environment")
        return count