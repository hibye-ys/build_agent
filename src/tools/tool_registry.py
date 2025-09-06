"""Tool registry for managing available tools."""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from langchain_core.tools import Tool, BaseTool

from .basic_tools import (
    get_calculator_tool,
    get_weather_tool,
    get_datetime_tool,
    get_file_tools,
    get_web_search_tool
)


logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing and accessing tools with MCP integration."""
    
    def __init__(self, mcp_manager=None):
        """Initialize the tool registry.
        
        Args:
            mcp_manager: Optional MCP manager instance for MCP tool integration
        """
        self._tools: Dict[str, BaseTool] = {}
        self._mcp_tools: Dict[str, BaseTool] = {}  # Separate MCP tools cache
        self._mcp_manager = mcp_manager
        self._load_default_tools()
    
    def _load_default_tools(self):
        """Load default tools into the registry."""
        # Basic tools
        self.register_tool("calculator", get_calculator_tool())
        self.register_tool("weather", get_weather_tool())
        self.register_tool("datetime", get_datetime_tool())
        self.register_tool("web_search", get_web_search_tool())
        
        # File tools
        file_tools = get_file_tools()
        for tool in file_tools:
            self.register_tool(tool.name, tool)
    
    def register_tool(self, name: str, tool: BaseTool):
        """Register a new tool.
        
        Args:
            name: Name identifier for the tool
            tool: Tool instance
        """
        self._tools[name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_tools(self, names: Optional[List[str]] = None) -> List[BaseTool]:
        """Get multiple tools by names.
        
        Args:
            names: List of tool names. If None, returns all tools.
            
        Returns:
            List of tool instances
        """
        if names is None:
            return list(self._tools.values())
        
        tools = []
        for name in names:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)
        return tools
    
    def list_tools(self) -> List[str]:
        """List all available tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry.
        
        Args:
            name: Tool name
            
        Returns:
            True if removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def set_mcp_manager(self, mcp_manager):
        """Set or update the MCP manager.
        
        Args:
            mcp_manager: MCP manager instance
        """
        self._mcp_manager = mcp_manager
        self._mcp_tools.clear()  # Clear cache when manager changes
    
    async def load_mcp_tools(
        self,
        server_name: Optional[str] = None,
        filter_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
        refresh: bool = False
    ) -> List[str]:
        """Load MCP tools into the registry.
        
        Args:
            server_name: If provided, only load from this server
            filter_tools: If provided, only include these tools
            exclude_tools: If provided, exclude these tools
            refresh: Whether to refresh the MCP tool cache
            
        Returns:
            List of loaded MCP tool names
        """
        if not self._mcp_manager:
            logger.warning("No MCP manager configured")
            return []
        
        try:
            # Get MCP tools
            mcp_tools = await self._mcp_manager.get_langchain_tools(
                server_name=server_name,
                filter_tools=filter_tools,
                exclude_tools=exclude_tools,
                refresh=refresh
            )
            
            # Clear existing MCP tools if refreshing
            if refresh:
                self._mcp_tools.clear()
            
            # Register MCP tools with prefixed names to avoid conflicts
            loaded_names = []
            for tool in mcp_tools:
                # Use original name for MCP tools, but track them separately
                tool_name = tool.name
                self._mcp_tools[tool_name] = tool
                loaded_names.append(tool_name)
                logger.debug(f"Loaded MCP tool: {tool_name}")
            
            logger.info(f"Loaded {len(loaded_names)} MCP tools")
            return loaded_names
            
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            return []
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name (checks both regular and MCP tools).
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        # Check regular tools first
        tool = self._tools.get(name)
        if tool:
            return tool
        
        # Check MCP tools
        return self._mcp_tools.get(name)
    
    def get_tools(self, names: Optional[List[str]] = None) -> List[BaseTool]:
        """Get multiple tools by names (includes both regular and MCP tools).
        
        Args:
            names: List of tool names. If None, returns all tools.
            
        Returns:
            List of tool instances
        """
        if names is None:
            # Return all tools (regular + MCP)
            all_tools = list(self._tools.values())
            all_tools.extend(self._mcp_tools.values())
            return all_tools
        
        tools = []
        for name in names:
            tool = self.get_tool(name)  # This checks both regular and MCP
            if tool:
                tools.append(tool)
        return tools
    
    def list_tools(self, include_mcp: bool = True) -> List[str]:
        """List all available tool names.
        
        Args:
            include_mcp: Whether to include MCP tools
            
        Returns:
            List of tool names
        """
        names = list(self._tools.keys())
        if include_mcp:
            names.extend(self._mcp_tools.keys())
        return names
    
    def list_mcp_tools(self) -> List[str]:
        """List only MCP tool names.
        
        Returns:
            List of MCP tool names
        """
        return list(self._mcp_tools.keys())
    
    def clear_mcp_tools(self):
        """Clear all MCP tools from the registry."""
        self._mcp_tools.clear()
        logger.info("Cleared all MCP tools from registry")


# Global registry instance
_global_registry = ToolRegistry()


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """Get a tool by name from the global registry.
    
    Args:
        name: Tool name
        
    Returns:
        Tool instance or None
    """
    return _global_registry.get_tool(name)


def get_tools_by_names(names: Union[str, List[str]]) -> List[BaseTool]:
    """Get tools by names from the global registry.
    
    Args:
        names: Tool name(s)
        
    Returns:
        List of tool instances
    """
    if isinstance(names, str):
        names = [names]
    return _global_registry.get_tools(names)


def get_all_tools() -> List[BaseTool]:
    """Get all available tools from the global registry.
    
    Returns:
        List of all tool instances
    """
    return _global_registry.get_tools()


def list_available_tools() -> List[str]:
    """List all available tool names.
    
    Returns:
        List of tool names
    """
    return _global_registry.list_tools()


def register_custom_tool(name: str, tool: BaseTool):
    """Register a custom tool in the global registry.
    
    Args:
        name: Tool name
        tool: Tool instance
    """
    _global_registry.register_tool(name, tool)


def set_mcp_manager(mcp_manager):
    """Set the MCP manager for the global registry.
    
    Args:
        mcp_manager: MCP manager instance
    """
    _global_registry.set_mcp_manager(mcp_manager)


async def load_mcp_tools(
    server_name: Optional[str] = None,
    filter_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
    refresh: bool = False
) -> List[str]:
    """Load MCP tools into the global registry.
    
    Args:
        server_name: If provided, only load from this server
        filter_tools: If provided, only include these tools  
        exclude_tools: If provided, exclude these tools
        refresh: Whether to refresh the MCP tool cache
        
    Returns:
        List of loaded MCP tool names
    """
    return await _global_registry.load_mcp_tools(
        server_name=server_name,
        filter_tools=filter_tools,
        exclude_tools=exclude_tools,
        refresh=refresh
    )


def list_mcp_tools() -> List[str]:
    """List MCP tool names from the global registry.
    
    Returns:
        List of MCP tool names
    """
    return _global_registry.list_mcp_tools()


def clear_mcp_tools():
    """Clear all MCP tools from the global registry."""
    _global_registry.clear_mcp_tools()