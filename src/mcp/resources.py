"""
Resource management for MCP integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json

from .client import MCPClient, MCPResource
from .registry import MCPRegistry
from .exceptions import MCPResourceError, MCPServerNotFoundError


logger = logging.getLogger(__name__)


@dataclass
class CachedResource:
    """Cached MCP resource."""
    uri: str
    server_name: str
    content: Any
    mime_type: Optional[str]
    cached_at: datetime
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl <= 0:
            return False  # Never expires
        return datetime.now() > self.cached_at + timedelta(seconds=self.ttl)
    
    def access(self):
        """Record an access to this resource."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class MCPResourceManager:
    """Manager for MCP resources across multiple servers."""
    
    def __init__(
        self,
        registry: Optional[MCPRegistry] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        max_cache_size: int = 1000
    ):
        """Initialize resource manager.
        
        Args:
            registry: MCP registry instance
            cache_enabled: Whether to enable caching
            cache_ttl: Default cache TTL in seconds
            max_cache_size: Maximum number of cached resources
        """
        self.registry = registry or MCPRegistry()
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, CachedResource] = {}
        self._resource_index: Dict[str, List[str]] = {}  # URI to server names mapping
    
    def _get_cache_key(self, server_name: str, uri: str) -> str:
        """Generate cache key for a resource.
        
        Args:
            server_name: Server name
            uri: Resource URI
            
        Returns:
            Cache key
        """
        return f"{server_name}:{uri}"
    
    async def discover_resources(self, server_name: Optional[str] = None) -> Dict[str, List[MCPResource]]:
        """Discover available resources from servers.
        
        Args:
            server_name: If provided, only discover from this server
            
        Returns:
            Dictionary mapping server names to their resources
        """
        discovered = {}
        
        if server_name:
            # Discover from specific server
            client = self.registry.get_client(server_name)
            if not client:
                raise MCPServerNotFoundError(server_name, self.registry.list_servers())
            
            if not client.is_connected:
                await client.connect()
            
            discovered[server_name] = client.resources
            
            # Update index
            for resource in client.resources:
                if resource.uri not in self._resource_index:
                    self._resource_index[resource.uri] = []
                if server_name not in self._resource_index[resource.uri]:
                    self._resource_index[resource.uri].append(server_name)
        else:
            # Discover from all connected servers
            for name in self.registry.list_servers():
                client = self.registry.get_client(name)
                if client and client.is_connected:
                    discovered[name] = client.resources
                    
                    # Update index
                    for resource in client.resources:
                        if resource.uri not in self._resource_index:
                            self._resource_index[resource.uri] = []
                        if name not in self._resource_index[resource.uri]:
                            self._resource_index[resource.uri].append(name)
        
        return discovered
    
    async def read_resource(
        self,
        uri: str,
        server_name: Optional[str] = None,
        use_cache: Optional[bool] = None,
        cache_ttl: Optional[int] = None
    ) -> Any:
        """Read a resource from MCP server.
        
        Args:
            uri: Resource URI
            server_name: Server to read from (auto-selects if not provided)
            use_cache: Whether to use cache (overrides default)
            cache_ttl: Cache TTL for this resource (overrides default)
            
        Returns:
            Resource content
        """
        use_cache = use_cache if use_cache is not None else self.cache_enabled
        cache_ttl = cache_ttl if cache_ttl is not None else self.cache_ttl
        
        # Determine server
        if not server_name:
            # Check if we know which servers have this resource
            if uri in self._resource_index and self._resource_index[uri]:
                # Use first available server
                for candidate in self._resource_index[uri]:
                    if candidate in self.registry.get_healthy_servers():
                        server_name = candidate
                        break
            
            if not server_name:
                # Try to find any server with this resource
                await self.discover_resources()
                if uri in self._resource_index and self._resource_index[uri]:
                    server_name = self._resource_index[uri][0]
                else:
                    raise MCPResourceError(uri, "No server found with this resource")
        
        # Check cache
        cache_key = self._get_cache_key(server_name, uri)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired:
                cached.access()
                logger.debug(f"Returning cached resource '{uri}' from '{server_name}'")
                return cached.content
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        # Get client
        client = self.registry.get_client(server_name)
        if not client:
            raise MCPServerNotFoundError(server_name, self.registry.list_servers())
        
        if not client.is_connected:
            await client.connect()
        
        try:
            # Read resource
            content = await client.read_resource(uri)
            
            # Find resource metadata
            resource_meta = next(
                (r for r in client.resources if r.uri == uri),
                None
            )
            
            # Cache if enabled
            if use_cache:
                self._add_to_cache(
                    cache_key,
                    CachedResource(
                        uri=uri,
                        server_name=server_name,
                        content=content,
                        mime_type=resource_meta.mime_type if resource_meta else None,
                        cached_at=datetime.now(),
                        ttl=cache_ttl
                    )
                )
            
            # Update server stats
            server_info = self.registry.get_server(server_name)
            if server_info:
                server_info.record_call(success=True)
            
            return content
            
        except Exception as e:
            # Update server stats
            server_info = self.registry.get_server(server_name)
            if server_info:
                server_info.record_call(success=False)
            
            raise MCPResourceError(uri, f"Failed to read resource: {e}")
    
    def _add_to_cache(self, key: str, resource: CachedResource):
        """Add resource to cache.
        
        Args:
            key: Cache key
            resource: Cached resource
        """
        # Check cache size limit
        if len(self._cache) >= self.max_cache_size:
            # Remove least recently accessed item
            lru_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed or self._cache[k].cached_at
            )
            del self._cache[lru_key]
        
        self._cache[key] = resource
        resource.access()
    
    async def read_resources(
        self,
        uris: List[str],
        parallel: bool = True,
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Read multiple resources.
        
        Args:
            uris: List of resource URIs
            parallel: Whether to read in parallel
            use_cache: Whether to use cache
            
        Returns:
            Dictionary mapping URIs to their content
        """
        if parallel:
            # Read in parallel
            tasks = {
                uri: asyncio.create_task(self.read_resource(uri, use_cache=use_cache))
                for uri in uris
            }
            
            results = {}
            for uri, task in tasks.items():
                try:
                    results[uri] = await task
                except Exception as e:
                    logger.error(f"Failed to read resource '{uri}': {e}")
                    results[uri] = None
        else:
            # Read sequentially
            results = {}
            for uri in uris:
                try:
                    results[uri] = await self.read_resource(uri, use_cache=use_cache)
                except Exception as e:
                    logger.error(f"Failed to read resource '{uri}': {e}")
                    results[uri] = None
        
        return results
    
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
        resources = []
        
        if server_name:
            # List from specific server
            client = self.registry.get_client(server_name)
            if client and client.is_connected:
                for resource in client.resources:
                    if mime_type and resource.mime_type != mime_type:
                        continue
                    
                    resources.append({
                        'uri': resource.uri,
                        'name': resource.name,
                        'description': resource.description,
                        'mime_type': resource.mime_type,
                        'server': server_name,
                        'metadata': resource.metadata
                    })
        else:
            # List from all servers
            for name in self.registry.list_servers():
                client = self.registry.get_client(name)
                if client and client.is_connected:
                    for resource in client.resources:
                        if mime_type and resource.mime_type != mime_type:
                            continue
                        
                        resources.append({
                            'uri': resource.uri,
                            'name': resource.name,
                            'description': resource.description,
                            'mime_type': resource.mime_type,
                            'server': name,
                            'metadata': resource.metadata
                        })
        
        return resources
    
    def search_resources(
        self,
        query: str,
        search_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for resources.
        
        Args:
            query: Search query
            search_fields: Fields to search in (uri, name, description)
            
        Returns:
            List of matching resources
        """
        if not search_fields:
            search_fields = ['uri', 'name', 'description']
        
        query_lower = query.lower()
        matching = []
        
        for resource_info in self.list_resources():
            # Check if query matches any field
            for field in search_fields:
                value = resource_info.get(field, '')
                if value and query_lower in str(value).lower():
                    matching.append(resource_info)
                    break
        
        return matching
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        if not self.cache_enabled:
            return {'enabled': False}
        
        total_size = len(self._cache)
        expired = sum(1 for r in self._cache.values() if r.is_expired)
        total_accesses = sum(r.access_count for r in self._cache.values())
        
        return {
            'enabled': True,
            'total_entries': total_size,
            'expired_entries': expired,
            'active_entries': total_size - expired,
            'total_accesses': total_accesses,
            'average_accesses': total_accesses / total_size if total_size > 0 else 0,
            'cache_hit_rate': self._calculate_hit_rate(),
            'max_size': self.max_cache_size,
            'ttl': self.cache_ttl
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate.
        
        Returns:
            Hit rate (0.0 to 1.0)
        """
        if not self._cache:
            return 0.0
        
        total_accesses = sum(r.access_count for r in self._cache.values())
        if total_accesses == 0:
            return 0.0
        
        # Assume first access is a miss, rest are hits
        hits = sum(max(0, r.access_count - 1) for r in self._cache.values())
        return hits / total_accesses
    
    def clear_cache(self, server_name: Optional[str] = None, uri: Optional[str] = None):
        """Clear resource cache.
        
        Args:
            server_name: If provided, only clear cache for this server
            uri: If provided, only clear cache for this URI
        """
        if uri:
            # Clear specific URI from all servers or specific server
            keys_to_remove = []
            for key in self._cache:
                cached = self._cache[key]
                if cached.uri == uri:
                    if not server_name or cached.server_name == server_name:
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
        elif server_name:
            # Clear all resources from specific server
            keys_to_remove = [
                key for key, cached in self._cache.items()
                if cached.server_name == server_name
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
        else:
            # Clear entire cache
            self._cache.clear()
    
    async def prefetch_resources(
        self,
        uris: List[str],
        cache_ttl: Optional[int] = None
    ) -> Dict[str, bool]:
        """Prefetch resources into cache.
        
        Args:
            uris: List of resource URIs to prefetch
            cache_ttl: Cache TTL for prefetched resources
            
        Returns:
            Dictionary mapping URIs to prefetch success
        """
        results = {}
        
        for uri in uris:
            try:
                await self.read_resource(uri, use_cache=True, cache_ttl=cache_ttl)
                results[uri] = True
            except Exception as e:
                logger.error(f"Failed to prefetch resource '{uri}': {e}")
                results[uri] = False
        
        return results