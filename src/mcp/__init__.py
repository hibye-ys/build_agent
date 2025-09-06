"""
MCP (Model Context Protocol) Integration Module for LangChain/LangGraph

This module provides a flexible and reusable system for integrating MCP servers
with LangChain and LangGraph agents, enabling seamless tool discovery and usage.
"""

from .client import MCPClient, MCPConnection
from .adapter import MCPLangChainAdapter, MCPToolConverter
from .config import MCPConfig, ServerConfig, ServerType
from .registry import MCPRegistry, MCPServerInfo
from .manager import MCPManager
from .resources import MCPResourceManager
from .exceptions import (
    MCPException,
    MCPConnectionError,
    MCPConfigurationError,
    MCPToolConversionError,
    MCPServerNotFoundError
)

__all__ = [
    # Main interface
    'MCPManager',
    
    # Core components
    'MCPClient',
    'MCPConnection',
    'MCPLangChainAdapter',
    'MCPToolConverter',
    'MCPConfig',
    'ServerConfig',
    'ServerType',
    'MCPRegistry',
    'MCPServerInfo',
    'MCPResourceManager',
    
    # Exceptions
    'MCPException',
    'MCPConnectionError',
    'MCPConfigurationError',
    'MCPToolConversionError',
    'MCPServerNotFoundError',
]

# Version info
__version__ = '1.0.0'