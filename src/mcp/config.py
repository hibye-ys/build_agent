"""
Configuration management for MCP integration.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from .exceptions import MCPConfigurationError


class ServerType(Enum):
    """MCP server connection types."""
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"
    SSE = "sse"


class AuthType(Enum):
    """Authentication types for MCP servers."""
    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH = "oauth"
    CUSTOM = "custom"


@dataclass
class AuthConfig:
    """Authentication configuration for MCP server."""
    type: AuthType = AuthType.NONE
    token: Optional[str] = None
    api_key: Optional[str] = None
    oauth_config: Optional[Dict[str, Any]] = None
    custom_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthConfig':
        """Create AuthConfig from dictionary."""
        if not data:
            return cls()
        
        auth_type = AuthType(data.get('type', 'none'))
        return cls(
            type=auth_type,
            token=data.get('token'),
            api_key=data.get('api_key'),
            oauth_config=data.get('oauth_config'),
            custom_config=data.get('custom_config')
        )
    
    def resolve_env_vars(self):
        """Resolve environment variables in configuration."""
        if self.token and self.token.startswith('${') and self.token.endswith('}'):
            env_var = self.token[2:-1]
            self.token = os.environ.get(env_var)
            if not self.token:
                raise MCPConfigurationError(f"Environment variable '{env_var}' not found")
        
        if self.api_key and self.api_key.startswith('${') and self.api_key.endswith('}'):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var)
            if not self.api_key:
                raise MCPConfigurationError(f"Environment variable '{env_var}' not found")


@dataclass
class ServerConfig:
    """Configuration for a single MCP server."""
    name: str
    type: ServerType
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    auth: AuthConfig = field(default_factory=AuthConfig)
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    capabilities: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'ServerConfig':
        """Create ServerConfig from dictionary."""
        server_type = ServerType(data.get('type', 'stdio'))
        
        # Validate required fields based on server type
        if server_type == ServerType.STDIO:
            if not data.get('command'):
                raise MCPConfigurationError(f"Server '{name}' with type 'stdio' requires 'command' field")
        elif server_type in [ServerType.HTTP, ServerType.WEBSOCKET]:
            if not data.get('url'):
                raise MCPConfigurationError(f"Server '{name}' with type '{server_type.value}' requires 'url' field")
        
        config = cls(
            name=name,
            type=server_type,
            command=data.get('command'),
            args=data.get('args', []),
            url=data.get('url'),
            env=data.get('env', {}),
            auth=AuthConfig.from_dict(data.get('auth', {})),
            timeout=data.get('timeout', 30),
            retry_attempts=data.get('retry_attempts', 3),
            retry_delay=data.get('retry_delay', 1.0),
            capabilities=data.get('capabilities'),
            metadata=data.get('metadata', {})
        )
        
        # Resolve environment variables
        config.auth.resolve_env_vars()
        
        # Resolve env vars in environment variables
        if config.env:
            for key, value in config.env.items():
                if value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    config.env[key] = os.environ.get(env_var, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['type'] = self.type.value
        data['auth']['type'] = self.auth.type.value
        return data


@dataclass
class MCPSettings:
    """Global MCP settings."""
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    lazy_loading: bool = True
    tool_filtering: Optional[Dict[str, List[str]]] = None
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_connections: int = 10
    health_check_interval: int = 60
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPSettings':
        """Create MCPSettings from dictionary."""
        return cls(
            connection_timeout=data.get('connection_timeout', 30),
            retry_attempts=data.get('retry_attempts', 3),
            retry_delay=data.get('retry_delay', 1.0),
            lazy_loading=data.get('lazy_loading', True),
            tool_filtering=data.get('tool_filtering'),
            cache_enabled=data.get('cache_enabled', True),
            cache_ttl=data.get('cache_ttl', 3600),
            max_connections=data.get('max_connections', 10),
            health_check_interval=data.get('health_check_interval', 60),
            log_level=data.get('log_level', 'INFO')
        )


class MCPConfig:
    """Main configuration manager for MCP integration."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize MCP configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.servers: Dict[str, ServerConfig] = {}
        self.settings: MCPSettings = MCPSettings()
        self.config_path = config_path
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: Union[str, Path]):
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise MCPConfigurationError(f"Configuration file not found", str(config_path))
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    data = json.load(f)
                else:
                    raise MCPConfigurationError(f"Unsupported config file format: {config_path.suffix}", str(config_path))
            
            self._load_from_dict(data)
            self.config_path = config_path
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise MCPConfigurationError(f"Failed to parse configuration file: {e}", str(config_path))
    
    def load_from_dict(self, data: Dict[str, Any]):
        """Load configuration from dictionary.
        
        Args:
            data: Configuration dictionary
        """
        self._load_from_dict(data)
    
    def _load_from_dict(self, data: Dict[str, Any]):
        """Internal method to load configuration from dictionary."""
        # Load settings
        if 'settings' in data:
            self.settings = MCPSettings.from_dict(data['settings'])
        
        # Load servers
        if 'servers' in data:
            for server_name, server_data in data['servers'].items():
                self.servers[server_name] = ServerConfig.from_dict(server_name, server_data)
    
    def add_server(self, config: ServerConfig):
        """Add a server configuration.
        
        Args:
            config: Server configuration
        """
        self.servers[config.name] = config
    
    def remove_server(self, name: str) -> bool:
        """Remove a server configuration.
        
        Args:
            name: Server name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.servers:
            del self.servers[name]
            return True
        return False
    
    def get_server(self, name: str) -> Optional[ServerConfig]:
        """Get server configuration by name.
        
        Args:
            name: Server name
            
        Returns:
            Server configuration or None
        """
        return self.servers.get(name)
    
    def list_servers(self) -> List[str]:
        """List all configured server names.
        
        Returns:
            List of server names
        """
        return list(self.servers.keys())
    
    def save_to_file(self, config_path: Optional[Union[str, Path]] = None):
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration (uses loaded path if not specified)
        """
        if not config_path and not self.config_path:
            raise MCPConfigurationError("No config path specified")
        
        save_path = Path(config_path or self.config_path)
        
        data = {
            'settings': asdict(self.settings),
            'servers': {name: server.to_dict() for name, server in self.servers.items()}
        }
        
        try:
            with open(save_path, 'w') as f:
                if save_path.suffix in ['.yaml', '.yml']:
                    yaml.safe_dump(data, f, default_flow_style=False)
                elif save_path.suffix == '.json':
                    json.dump(data, f, indent=2)
                else:
                    raise MCPConfigurationError(f"Unsupported config file format: {save_path.suffix}")
        except Exception as e:
            raise MCPConfigurationError(f"Failed to save configuration: {e}", str(save_path))
    
    def validate(self) -> List[str]:
        """Validate configuration.
        
        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []
        
        # Check for duplicate servers
        if not self.servers:
            warnings.append("No MCP servers configured")
        
        # Validate each server
        for name, server in self.servers.items():
            if server.type == ServerType.STDIO and not server.command:
                warnings.append(f"Server '{name}': STDIO server missing command")
            elif server.type in [ServerType.HTTP, ServerType.WEBSOCKET] and not server.url:
                warnings.append(f"Server '{name}': {server.type.value} server missing URL")
            
            if server.timeout <= 0:
                warnings.append(f"Server '{name}': Invalid timeout {server.timeout}")
            
            if server.retry_attempts < 0:
                warnings.append(f"Server '{name}': Invalid retry attempts {server.retry_attempts}")
        
        # Validate settings
        if self.settings.connection_timeout <= 0:
            warnings.append(f"Invalid connection timeout: {self.settings.connection_timeout}")
        
        if self.settings.max_connections <= 0:
            warnings.append(f"Invalid max connections: {self.settings.max_connections}")
        
        return warnings