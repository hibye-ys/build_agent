"""
Registry for managing multiple MCP servers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .client import MCPClient, ConnectionState
from .config import ServerConfig
from .exceptions import MCPServerNotFoundError, MCPConnectionError


logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """Status of an MCP server."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"


@dataclass
class MCPServerInfo:
    """Information about an MCP server."""
    name: str
    config: ServerConfig
    client: Optional[MCPClient] = None
    status: ServerStatus = ServerStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    total_calls: int = 0
    successful_calls: int = 0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tool_count: int = 0
    resource_count: int = 0
    prompt_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def is_healthy(self) -> bool:
        """Check if server is healthy."""
        return self.status == ServerStatus.HEALTHY
    
    def update_status(self, status: ServerStatus):
        """Update server status."""
        self.status = status
        self.last_health_check = datetime.now()
    
    def record_call(self, success: bool):
        """Record a tool call result."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.error_count += 1


class MCPRegistry:
    """Registry for managing multiple MCP servers."""
    
    def __init__(
        self,
        health_check_interval: int = 60,
        auto_connect: bool = False,
        max_concurrent_connections: int = 10
    ):
        """Initialize MCP registry.
        
        Args:
            health_check_interval: Interval between health checks in seconds
            auto_connect: Whether to automatically connect to servers when registered
            max_concurrent_connections: Maximum number of concurrent connections
        """
        self.servers: Dict[str, MCPServerInfo] = {}
        self.health_check_interval = health_check_interval
        self.auto_connect = auto_connect
        self.max_concurrent_connections = max_concurrent_connections
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_semaphore = asyncio.Semaphore(max_concurrent_connections)
        self._tags: Dict[str, Set[str]] = {}  # Tag to server names mapping
    
    async def register_server(
        self,
        config: ServerConfig,
        connect: bool = None,
        tags: Optional[List[str]] = None
    ) -> MCPServerInfo:
        """Register an MCP server.
        
        Args:
            config: Server configuration
            connect: Whether to connect immediately (overrides auto_connect)
            tags: Optional tags for the server
            
        Returns:
            Server information
        """
        # Create server info
        server_info = MCPServerInfo(
            name=config.name,
            config=config,
            metadata=config.metadata or {}
        )
        
        # Add to registry
        self.servers[config.name] = server_info
        
        # Add tags
        if tags:
            for tag in tags:
                if tag not in self._tags:
                    self._tags[tag] = set()
                self._tags[tag].add(config.name)
        
        # Connect if requested
        should_connect = connect if connect is not None else self.auto_connect
        if should_connect:
            await self.connect_server(config.name)
        
        logger.info(f"Registered MCP server '{config.name}'")
        return server_info
    
    async def unregister_server(self, name: str) -> bool:
        """Unregister an MCP server.
        
        Args:
            name: Server name
            
        Returns:
            True if unregistered, False if not found
        """
        if name not in self.servers:
            return False
        
        # Disconnect if connected
        server_info = self.servers[name]
        if server_info.client:
            await server_info.client.disconnect()
        
        # Remove from tags
        for tag_servers in self._tags.values():
            tag_servers.discard(name)
        
        # Remove from registry
        del self.servers[name]
        
        logger.info(f"Unregistered MCP server '{name}'")
        return True
    
    async def connect_server(self, name: str) -> bool:
        """Connect to an MCP server.
        
        Args:
            name: Server name
            
        Returns:
            True if connected successfully
        """
        server_info = self.servers.get(name)
        if not server_info:
            raise MCPServerNotFoundError(name, list(self.servers.keys()))
        
        # Check if already connected
        if server_info.client and server_info.client.is_connected:
            return True
        
        async with self._connection_semaphore:
            try:
                # Update status
                server_info.update_status(ServerStatus.CONNECTING)
                
                # Create client if needed
                if not server_info.client:
                    server_info.client = MCPClient(server_info.config)
                
                # Connect
                success = await server_info.client.connect()
                
                if success:
                    # Update server info
                    server_info.update_status(ServerStatus.HEALTHY)
                    server_info.tool_count = len(server_info.client.tools)
                    server_info.resource_count = len(server_info.client.resources)
                    server_info.prompt_count = len(server_info.client.prompts)
                    
                    logger.info(f"Connected to MCP server '{name}' with {server_info.tool_count} tools")
                else:
                    server_info.update_status(ServerStatus.UNHEALTHY)
                    server_info.error_count += 1
                
                return success
                
            except Exception as e:
                server_info.update_status(ServerStatus.UNHEALTHY)
                server_info.error_count += 1
                logger.error(f"Failed to connect to MCP server '{name}': {e}")
                return False
    
    async def disconnect_server(self, name: str) -> bool:
        """Disconnect from an MCP server.
        
        Args:
            name: Server name
            
        Returns:
            True if disconnected successfully
        """
        server_info = self.servers.get(name)
        if not server_info:
            raise MCPServerNotFoundError(name, list(self.servers.keys()))
        
        if server_info.client:
            await server_info.client.disconnect()
            server_info.update_status(ServerStatus.DISCONNECTED)
            logger.info(f"Disconnected from MCP server '{name}'")
            return True
        
        return False
    
    async def connect_all(self, parallel: bool = True) -> Dict[str, bool]:
        """Connect to all registered servers.
        
        Args:
            parallel: Whether to connect in parallel
            
        Returns:
            Dictionary mapping server names to connection results
        """
        results = {}
        
        if parallel:
            # Connect in parallel
            tasks = {
                name: asyncio.create_task(self.connect_server(name))
                for name in self.servers
            }
            
            for name, task in tasks.items():
                try:
                    results[name] = await task
                except Exception as e:
                    logger.error(f"Failed to connect to '{name}': {e}")
                    results[name] = False
        else:
            # Connect sequentially
            for name in self.servers:
                try:
                    results[name] = await self.connect_server(name)
                except Exception as e:
                    logger.error(f"Failed to connect to '{name}': {e}")
                    results[name] = False
        
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect from all servers.
        
        Returns:
            Dictionary mapping server names to disconnection results
        """
        results = {}
        
        for name in self.servers:
            try:
                results[name] = await self.disconnect_server(name)
            except Exception as e:
                logger.error(f"Failed to disconnect from '{name}': {e}")
                results[name] = False
        
        return results
    
    def get_server(self, name: str) -> Optional[MCPServerInfo]:
        """Get server information.
        
        Args:
            name: Server name
            
        Returns:
            Server information or None
        """
        return self.servers.get(name)
    
    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get MCP client for a server.
        
        Args:
            name: Server name
            
        Returns:
            MCP client or None
        """
        server_info = self.servers.get(name)
        return server_info.client if server_info else None
    
    def list_servers(
        self,
        status: Optional[ServerStatus] = None,
        tag: Optional[str] = None
    ) -> List[str]:
        """List server names.
        
        Args:
            status: Filter by status
            tag: Filter by tag
            
        Returns:
            List of server names
        """
        names = set(self.servers.keys())
        
        # Filter by tag
        if tag and tag in self._tags:
            names &= self._tags[tag]
        
        # Filter by status
        if status:
            names = {
                name for name in names
                if self.servers[name].status == status
            }
        
        return sorted(names)
    
    def get_healthy_servers(self) -> List[str]:
        """Get list of healthy server names.
        
        Returns:
            List of healthy server names
        """
        return self.list_servers(status=ServerStatus.HEALTHY)
    
    def get_server_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a server.
        
        Args:
            name: Server name
            
        Returns:
            Server statistics or None
        """
        server_info = self.servers.get(name)
        if not server_info:
            return None
        
        return {
            'name': server_info.name,
            'status': server_info.status.value,
            'last_health_check': server_info.last_health_check.isoformat() if server_info.last_health_check else None,
            'error_count': server_info.error_count,
            'total_calls': server_info.total_calls,
            'successful_calls': server_info.successful_calls,
            'success_rate': server_info.success_rate,
            'tool_count': server_info.tool_count,
            'resource_count': server_info.resource_count,
            'prompt_count': server_info.prompt_count
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all servers.
        
        Returns:
            Dictionary mapping server names to their statistics
        """
        return {
            name: self.get_server_stats(name)
            for name in self.servers
        }
    
    async def health_check(self, name: str) -> bool:
        """Perform health check on a server.
        
        Args:
            name: Server name
            
        Returns:
            True if healthy
        """
        server_info = self.servers.get(name)
        if not server_info:
            return False
        
        # Check if client exists and is connected
        if not server_info.client:
            server_info.update_status(ServerStatus.DISCONNECTED)
            return False
        
        try:
            # Perform health check
            is_healthy = await server_info.client.connection.health_check()
            
            # Update status
            if is_healthy:
                server_info.update_status(ServerStatus.HEALTHY)
            else:
                server_info.update_status(ServerStatus.UNHEALTHY)
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for '{name}': {e}")
            server_info.update_status(ServerStatus.UNHEALTHY)
            server_info.error_count += 1
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all servers.
        
        Returns:
            Dictionary mapping server names to health check results
        """
        tasks = {
            name: asyncio.create_task(self.health_check(name))
            for name in self.servers
        }
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Health check failed for '{name}': {e}")
                results[name] = False
        
        return results
    
    async def start_health_monitoring(self):
        """Start background health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped health monitoring")
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                # Wait for interval
                await asyncio.sleep(self.health_check_interval)
                
                # Perform health checks
                await self.health_check_all()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
    
    def select_server(
        self,
        strategy: str = "round_robin",
        tag: Optional[str] = None
    ) -> Optional[str]:
        """Select a server based on strategy.
        
        Args:
            strategy: Selection strategy (round_robin, least_loaded, best_success_rate)
            tag: Optional tag filter
            
        Returns:
            Selected server name or None
        """
        # Get candidate servers
        candidates = self.get_healthy_servers()
        
        # Filter by tag
        if tag and tag in self._tags:
            candidates = [s for s in candidates if s in self._tags[tag]]
        
        if not candidates:
            return None
        
        if strategy == "round_robin":
            # Simple round-robin (would need state tracking for true round-robin)
            return candidates[0]
        
        elif strategy == "least_loaded":
            # Select server with fewest total calls
            return min(candidates, key=lambda s: self.servers[s].total_calls)
        
        elif strategy == "best_success_rate":
            # Select server with best success rate
            return max(candidates, key=lambda s: self.servers[s].success_rate)
        
        else:
            # Default to first available
            return candidates[0]
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_health_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_health_monitoring()
        await self.disconnect_all()