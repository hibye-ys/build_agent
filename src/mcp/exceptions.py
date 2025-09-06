"""
Custom exceptions for MCP integration module.
"""

from typing import Optional, Any, Dict


class MCPException(Exception):
    """Base exception for all MCP-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class MCPConnectionError(MCPException):
    """Raised when connection to MCP server fails."""
    
    def __init__(self, server_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.server_name = server_name
        super().__init__(f"Failed to connect to MCP server '{server_name}': {message}", details)


class MCPConfigurationError(MCPException):
    """Raised when MCP configuration is invalid."""
    
    def __init__(self, message: str, config_path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.config_path = config_path
        if config_path:
            message = f"Configuration error in '{config_path}': {message}"
        super().__init__(message, details)


class MCPToolConversionError(MCPException):
    """Raised when tool conversion from MCP to LangChain fails."""
    
    def __init__(self, tool_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.tool_name = tool_name
        super().__init__(f"Failed to convert MCP tool '{tool_name}': {message}", details)


class MCPServerNotFoundError(MCPException):
    """Raised when requested MCP server is not found."""
    
    def __init__(self, server_name: str, available_servers: Optional[list] = None):
        self.server_name = server_name
        self.available_servers = available_servers or []
        message = f"MCP server '{server_name}' not found"
        if self.available_servers:
            message += f". Available servers: {', '.join(self.available_servers)}"
        super().__init__(message)


class MCPTimeoutError(MCPException):
    """Raised when MCP operation times out."""
    
    def __init__(self, operation: str, timeout: float, details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"MCP operation '{operation}' timed out after {timeout} seconds", details)


class MCPResourceError(MCPException):
    """Raised when MCP resource operation fails."""
    
    def __init__(self, resource_uri: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.resource_uri = resource_uri
        super().__init__(f"Resource error for '{resource_uri}': {message}", details)


class MCPAuthenticationError(MCPException):
    """Raised when MCP server authentication fails."""
    
    def __init__(self, server_name: str, message: str = "Authentication failed"):
        self.server_name = server_name
        super().__init__(f"Authentication failed for MCP server '{server_name}': {message}")