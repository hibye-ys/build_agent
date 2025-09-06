"""
Main MCP manager for coordinating all MCP components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from langchain.tools import BaseTool

from .config import MCPConfig, ServerConfig
from .registry import MCPRegistry, MCPServerInfo
from .adapter import MCPLangChainAdapter
from .resources import MCPResourceManager
from .exceptions import MCPException, MCPServerNotFoundError


logger = logging.getLogger(__name__)


class MCPManager:
    """Main manager for MCP integration with LangChain/LangGraph."""
    
    def __init__(
        self,
        config: Optional[MCPConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        auto_connect: bool = False,
        lazy_loading: bool = True,
        cache_enabled: bool = True
    ):
        """Initialize MCP manager.
        
        Args:
            config: MCP configuration object
            config_path: Path to configuration file
            auto_connect: Whether to automatically connect to servers
            lazy_loading: Whether to use lazy loading for tools
            cache_enabled: Whether to enable caching
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = MCPConfig(config_path)
        else:
            self.config = MCPConfig()
        
        # Initialize components
        self.registry = MCPRegistry(
            health_check_interval=self.config.settings.health_check_interval,
            auto_connect=auto_connect,
            max_concurrent_connections=self.config.settings.max_connections
        )
        
        self.adapter = MCPLangChainAdapter(
            handle_errors=True,
            lazy_loading=lazy_loading,
            cache_tools=cache_enabled
        )
        
        self.resource_manager = MCPResourceManager(
            registry=self.registry,
            cache_enabled=cache_enabled,
            cache_ttl=self.config.settings.cache_ttl,
            max_cache_size=1000
        )
        
        self._initialized = False
        self._langchain_tools: Dict[str, List[BaseTool]] = {}
    
    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        **kwargs
    ) -> 'MCPManager':
        """Create manager from configuration file.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments for manager
            
        Returns:
            MCP manager instance
        """
        return cls(config_path=config_path, **kwargs)
    
    async def initialize(self) -> bool:
        """Initialize the MCP manager.
        
        Returns:
            True if initialized successfully
        """
        if self._initialized:
            return True
        
        try:
            # Register all configured servers
            for server_name, server_config in self.config.servers.items():
                await self.registry.register_server(
                    server_config,
                    connect=False  # Don't connect yet
                )
            
            # Start health monitoring
            await self.registry.start_health_monitoring()
            
            self._initialized = True
            logger.info(f"MCP Manager initialized with {len(self.config.servers)} servers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Manager: {e}")
            return False
    
    async def connect(self, server_name: Optional[str] = None) -> Dict[str, bool]:
        """Connect to MCP servers.
        
        Args:
            server_name: If provided, only connect to this server
            
        Returns:
            Dictionary mapping server names to connection results
        """
        if not self._initialized:
            await self.initialize()
        
        if server_name:
            # Connect to specific server
            success = await self.registry.connect_server(server_name)
            return {server_name: success}
        else:
            # Connect to all servers
            return await self.registry.connect_all(parallel=True)
    
    async def disconnect(self, server_name: Optional[str] = None) -> Dict[str, bool]:
        """Disconnect from MCP servers.
        
        Args:
            server_name: If provided, only disconnect from this server
            
        Returns:
            Dictionary mapping server names to disconnection results
        """
        if server_name:
            # Disconnect from specific server
            success = await self.registry.disconnect_server(server_name)
            
            # Clear tool cache for this server
            if server_name in self._langchain_tools:
                del self._langchain_tools[server_name]
            self.adapter.clear_cache(server_name)
            
            return {server_name: success}
        else:
            # Disconnect from all servers
            result = await self.registry.disconnect_all()
            
            # Clear all tool caches
            self._langchain_tools.clear()
            self.adapter.clear_cache()
            
            return result
    
    async def get_langchain_tools(
        self,
        server_name: Optional[str] = None,
        filter_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
        refresh: bool = False
    ) -> List[BaseTool]:
        """Get LangChain tools from MCP servers.
        
        Args:
            server_name: If provided, only get tools from this server
            filter_tools: If provided, only include these tools
            exclude_tools: If provided, exclude these tools
            refresh: Whether to refresh the tool cache
            
        Returns:
            List of LangChain tool instances
        """
        if not self._initialized:
            await self.initialize()
        
        # Clear cache if refresh requested
        if refresh:
            if server_name:
                if server_name in self._langchain_tools:
                    del self._langchain_tools[server_name]
                self.adapter.clear_cache(server_name)
            else:
                self._langchain_tools.clear()
                self.adapter.clear_cache()
        
        all_tools = []
        
        if server_name:
            # Get tools from specific server
            if server_name not in self._langchain_tools or refresh:
                client = self.registry.get_client(server_name)
                if not client:
                    raise MCPServerNotFoundError(server_name, self.registry.list_servers())
                
                if not client.is_connected:
                    await client.connect()
                
                tools = await self.adapter.create_tools(
                    client,
                    filter_tools=filter_tools,
                    exclude_tools=exclude_tools
                )
                self._langchain_tools[server_name] = tools
            
            all_tools = self._langchain_tools[server_name]
        else:
            # Get tools from all connected servers
            for name in self.registry.get_healthy_servers():
                if name not in self._langchain_tools or refresh:
                    client = self.registry.get_client(name)
                    if client and client.is_connected:
                        tools = await self.adapter.create_tools(
                            client,
                            filter_tools=filter_tools,
                            exclude_tools=exclude_tools
                        )
                        self._langchain_tools[name] = tools
                
                if name in self._langchain_tools:
                    all_tools.extend(self._langchain_tools[name])
        
        # Apply global filtering if getting from all servers
        if not server_name:
            if filter_tools:
                all_tools = [t for t in all_tools if t.name in filter_tools]
            if exclude_tools:
                all_tools = [t for t in all_tools if t.name not in exclude_tools]
        
        return all_tools
    
    def list_tool_names(self, server_name: Optional[str] = None) -> List[str]:
        """List available tool names.
        
        Args:
            server_name: If provided, only list from this server
            
        Returns:
            List of tool names
        """
        tool_names = []
        
        if server_name:
            client = self.registry.get_client(server_name)
            if client and client.is_connected:
                tool_names = [tool.name for tool in client.tools]
        else:
            # List from all connected servers
            for name in self.registry.get_healthy_servers():
                client = self.registry.get_client(name)
                if client and client.is_connected:
                    tool_names.extend([tool.name for tool in client.tools])
        
        return list(set(tool_names))  # Remove duplicates
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None
    ) -> Any:
        """Call an MCP tool directly.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            server_name: Server to use (auto-selects if not provided)
            
        Returns:
            Tool result
        """
        if not server_name:
            # Find a server with this tool
            for name in self.registry.get_healthy_servers():
                client = self.registry.get_client(name)
                if client and any(t.name == tool_name for t in client.tools):
                    server_name = name
                    break
            
            if not server_name:
                raise MCPException(f"No server found with tool '{tool_name}'")
        
        client = self.registry.get_client(server_name)
        if not client:
            raise MCPServerNotFoundError(server_name, self.registry.list_servers())
        
        if not client.is_connected:
            await client.connect()
        
        return await client.call_tool(tool_name, arguments)
    
    async def read_resource(
        self,
        uri: str,
        server_name: Optional[str] = None,
        use_cache: Optional[bool] = None
    ) -> Any:
        """Read an MCP resource.
        
        Args:
            uri: Resource URI
            server_name: Server to read from (auto-selects if not provided)
            use_cache: Whether to use cache
            
        Returns:
            Resource content
        """
        return await self.resource_manager.read_resource(
            uri,
            server_name=server_name,
            use_cache=use_cache
        )
    
    def list_resources(
        self,
        server_name: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available resources.
        
        Args:
            server_name: Filter by server
            mime_type: Filter by MIME type
            
        Returns:
            List of resource information
        """
        return self.resource_manager.list_resources(
            server_name=server_name,
            mime_type=mime_type
        )
    
    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None
    ) -> str:
        """Get an MCP prompt.
        
        Args:
            prompt_name: Prompt name
            arguments: Prompt arguments
            server_name: Server to use (auto-selects if not provided)
            
        Returns:
            Expanded prompt text
        """
        if not server_name:
            # Find a server with this prompt
            for name in self.registry.get_healthy_servers():
                client = self.registry.get_client(name)
                if client and any(p.name == prompt_name for p in client.prompts):
                    server_name = name
                    break
            
            if not server_name:
                raise MCPException(f"No server found with prompt '{prompt_name}'")
        
        client = self.registry.get_client(server_name)
        if not client:
            raise MCPServerNotFoundError(server_name, self.registry.list_servers())
        
        if not client.is_connected:
            await client.connect()
        
        return await client.get_prompt(prompt_name, arguments)
    
    def add_server(
        self,
        server_config: ServerConfig,
        connect: bool = False
    ) -> asyncio.Task:
        """Add a new server dynamically.
        
        Args:
            server_config: Server configuration
            connect: Whether to connect immediately
            
        Returns:
            Task for server registration
        """
        # Add to config
        self.config.add_server(server_config)
        
        # Register with registry
        return asyncio.create_task(
            self.registry.register_server(server_config, connect=connect)
        )
    
    def remove_server(self, server_name: str) -> asyncio.Task:
        """Remove a server dynamically.
        
        Args:
            server_name: Server name
            
        Returns:
            Task for server removal
        """
        # Remove from config
        self.config.remove_server(server_name)
        
        # Clear tool cache
        if server_name in self._langchain_tools:
            del self._langchain_tools[server_name]
        self.adapter.clear_cache(server_name)
        
        # Unregister from registry
        return asyncio.create_task(
            self.registry.unregister_server(server_name)
        )
    
    def get_server_info(self, server_name: str) -> Optional[MCPServerInfo]:
        """Get information about a server.
        
        Args:
            server_name: Server name
            
        Returns:
            Server information or None
        """
        return self.registry.get_server(server_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics.
        
        Returns:
            Manager statistics
        """
        return {
            'servers': self.registry.get_all_stats(),
            'cache': self.resource_manager.get_cache_stats(),
            'total_servers': len(self.config.servers),
            'connected_servers': len(self.registry.get_healthy_servers()),
            'total_tools': sum(len(tools) for tools in self._langchain_tools.values()),
            'initialized': self._initialized
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all servers.
        
        Returns:
            Dictionary mapping server names to health status
        """
        return await self.registry.health_check_all()
    
    async def shutdown(self):
        """Shutdown the MCP manager."""
        # Stop health monitoring
        await self.registry.stop_health_monitoring()
        
        # Disconnect all servers
        await self.disconnect()
        
        # Clear caches
        self.adapter.clear_cache()
        self.resource_manager.clear_cache()
        
        self._initialized = False
        logger.info("MCP Manager shutdown complete")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()