"""Tool registry for managing available tools."""

from typing import Dict, List, Optional, Union
from langchain_core.tools import Tool, BaseTool

from .basic_tools import (
    get_calculator_tool,
    get_weather_tool,
    get_datetime_tool,
    get_file_tools,
    get_web_search_tool
)


class ToolRegistry:
    """Registry for managing and accessing tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
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