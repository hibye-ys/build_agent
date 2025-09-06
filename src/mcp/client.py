"""
MCP client implementation for connecting to MCP servers.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from .config import ServerConfig, ServerType
from .exceptions import (
    MCPConnectionError, 
    MCPTimeoutError,
    MCPAuthenticationError
)


logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state for MCP client."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'inputSchema': self.input_schema,
            'metadata': self.metadata or {}
        }


@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'uri': self.uri,
            'name': self.name,
            'description': self.description,
            'mimeType': self.mime_type,
            'metadata': self.metadata or {}
        }


@dataclass
class MCPPrompt:
    """Represents an MCP prompt template."""
    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPConnection(ABC):
    """Abstract base class for MCP connections."""
    
    def __init__(self, config: ServerConfig):
        """Initialize MCP connection.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self.prompts: List[MCPPrompt] = []
        self._callbacks: Dict[str, List[Callable]] = {}
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to MCP server.
        
        Returns:
            True if connected successfully
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from MCP server."""
        pass
    
    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        pass
    
    @abstractmethod
    async def read_resource(self, uri: str) -> Any:
        """Read an MCP resource.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource content
        """
        pass
    
    @abstractmethod
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Get an MCP prompt.
        
        Args:
            name: Prompt name
            arguments: Prompt arguments
            
        Returns:
            Expanded prompt text
        """
        pass
    
    def on_state_change(self, callback: Callable[[ConnectionState], None]):
        """Register state change callback.
        
        Args:
            callback: Callback function
        """
        if 'state_change' not in self._callbacks:
            self._callbacks['state_change'] = []
        self._callbacks['state_change'].append(callback)
    
    def _set_state(self, state: ConnectionState):
        """Set connection state and notify callbacks."""
        self.state = state
        for callback in self._callbacks.get('state_change', []):
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    async def health_check(self) -> bool:
        """Check if connection is healthy.
        
        Returns:
            True if healthy
        """
        return self.state == ConnectionState.CONNECTED


class StdioMCPConnection(MCPConnection):
    """STDIO-based MCP connection implementation."""
    
    def __init__(self, config: ServerConfig):
        """Initialize STDIO MCP connection."""
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self._read_task: Optional[asyncio.Task] = None
        self._message_id = 0
        self._pending_responses: Dict[int, asyncio.Future] = {}
    
    async def connect(self) -> bool:
        """Connect to MCP server via STDIO."""
        if self.state == ConnectionState.CONNECTED:
            return True
        
        self._set_state(ConnectionState.CONNECTING)
        
        try:
            # Build command
            cmd = [self.config.command] + (self.config.args or [])
            
            # Set up environment
            env = dict(os.environ)
            if self.config.env:
                env.update(self.config.env)
            
            # Start process
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Start reading responses
            self._read_task = asyncio.create_task(self._read_loop())
            
            # Initialize connection
            await self._initialize()
            
            self._set_state(ConnectionState.CONNECTED)
            logger.info(f"Connected to MCP server '{self.config.name}' via STDIO")
            return True
            
        except Exception as e:
            self._set_state(ConnectionState.FAILED)
            raise MCPConnectionError(self.config.name, str(e))
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.state == ConnectionState.DISCONNECTED:
            return
        
        self._set_state(ConnectionState.DISCONNECTED)
        
        # Cancel read task
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        # Terminate process
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.process = None
        
        logger.info(f"Disconnected from MCP server '{self.config.name}'")
    
    async def _initialize(self):
        """Initialize connection and discover capabilities."""
        # Send initialize request
        response = await self._send_request('initialize', {
            'protocolVersion': '0.1.0',
            'capabilities': {
                'tools': {},
                'resources': {},
                'prompts': {}
            }
        })
        
        # Store server capabilities
        capabilities = response.get('capabilities', {})
        
        # List tools
        if capabilities.get('tools'):
            tools_response = await self._send_request('tools/list', {})
            self.tools = [
                MCPTool(
                    name=tool['name'],
                    description=tool.get('description', ''),
                    input_schema=tool.get('inputSchema', {}),
                    metadata=tool.get('metadata')
                )
                for tool in tools_response.get('tools', [])
            ]
        
        # List resources
        if capabilities.get('resources'):
            resources_response = await self._send_request('resources/list', {})
            self.resources = [
                MCPResource(
                    uri=res['uri'],
                    name=res['name'],
                    description=res.get('description'),
                    mime_type=res.get('mimeType'),
                    metadata=res.get('metadata')
                )
                for res in resources_response.get('resources', [])
            ]
        
        # List prompts
        if capabilities.get('prompts'):
            prompts_response = await self._send_request('prompts/list', {})
            self.prompts = [
                MCPPrompt(
                    name=prompt['name'],
                    description=prompt.get('description'),
                    arguments=prompt.get('arguments'),
                    metadata=prompt.get('metadata')
                )
                for prompt in prompts_response.get('prompts', [])
            ]
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server.
        
        Args:
            method: Method name
            params: Method parameters
            
        Returns:
            Response data
        """
        if not self.process or not self.process.stdin:
            raise MCPConnectionError(self.config.name, "Not connected")
        
        # Create request
        self._message_id += 1
        request = {
            'jsonrpc': '2.0',
            'id': self._message_id,
            'method': method,
            'params': params
        }
        
        # Create future for response
        future = asyncio.Future()
        self._pending_responses[self._message_id] = future
        
        # Send request
        try:
            request_str = json.dumps(request) + '\n'
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=self.config.timeout)
            
            if 'error' in response:
                raise MCPConnectionError(
                    self.config.name,
                    f"MCP error: {response['error'].get('message', 'Unknown error')}",
                    response['error']
                )
            
            return response.get('result', {})
            
        except asyncio.TimeoutError:
            del self._pending_responses[self._message_id]
            raise MCPTimeoutError(method, self.config.timeout)
        except Exception as e:
            if self._message_id in self._pending_responses:
                del self._pending_responses[self._message_id]
            raise
    
    async def _read_loop(self):
        """Read responses from MCP server."""
        if not self.process or not self.process.stdout:
            return
        
        while True:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                # Parse response
                try:
                    response = json.loads(line.decode())
                    
                    # Handle response
                    if 'id' in response:
                        message_id = response['id']
                        if message_id in self._pending_responses:
                            self._pending_responses[message_id].set_result(response)
                    else:
                        # Handle notifications
                        await self._handle_notification(response)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse MCP response: {e}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                break
    
    async def _handle_notification(self, notification: Dict[str, Any]):
        """Handle MCP notification.
        
        Args:
            notification: Notification data
        """
        method = notification.get('method')
        params = notification.get('params', {})
        
        # Handle different notification types
        if method == 'tools/changed':
            # Refresh tools list
            await self._refresh_tools()
        elif method == 'resources/changed':
            # Refresh resources list
            await self._refresh_resources()
        elif method == 'prompts/changed':
            # Refresh prompts list
            await self._refresh_prompts()
        else:
            logger.debug(f"Received notification: {method}")
    
    async def _refresh_tools(self):
        """Refresh tools list."""
        try:
            response = await self._send_request('tools/list', {})
            self.tools = [
                MCPTool(
                    name=tool['name'],
                    description=tool.get('description', ''),
                    input_schema=tool.get('inputSchema', {}),
                    metadata=tool.get('metadata')
                )
                for tool in response.get('tools', [])
            ]
        except Exception as e:
            logger.error(f"Failed to refresh tools: {e}")
    
    async def _refresh_resources(self):
        """Refresh resources list."""
        try:
            response = await self._send_request('resources/list', {})
            self.resources = [
                MCPResource(
                    uri=res['uri'],
                    name=res['name'],
                    description=res.get('description'),
                    mime_type=res.get('mimeType'),
                    metadata=res.get('metadata')
                )
                for res in response.get('resources', [])
            ]
        except Exception as e:
            logger.error(f"Failed to refresh resources: {e}")
    
    async def _refresh_prompts(self):
        """Refresh prompts list."""
        try:
            response = await self._send_request('prompts/list', {})
            self.prompts = [
                MCPPrompt(
                    name=prompt['name'],
                    description=prompt.get('description'),
                    arguments=prompt.get('arguments'),
                    metadata=prompt.get('metadata')
                )
                for prompt in response.get('prompts', [])
            ]
        except Exception as e:
            logger.error(f"Failed to refresh prompts: {e}")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool."""
        response = await self._send_request('tools/call', {
            'name': name,
            'arguments': arguments
        })
        return response.get('content', [])
    
    async def read_resource(self, uri: str) -> Any:
        """Read an MCP resource."""
        response = await self._send_request('resources/read', {
            'uri': uri
        })
        return response.get('contents', [])
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Get an MCP prompt."""
        response = await self._send_request('prompts/get', {
            'name': name,
            'arguments': arguments or {}
        })
        
        # Combine prompt messages
        messages = response.get('messages', [])
        return '\n'.join(msg.get('content', '') for msg in messages)


class MCPClient:
    """Main MCP client for managing connections."""
    
    def __init__(self, config: ServerConfig):
        """Initialize MCP client.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.connection: Optional[MCPConnection] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
    
    async def connect(self) -> bool:
        """Connect to MCP server.
        
        Returns:
            True if connected successfully
        """
        # Create connection based on server type
        if self.config.type == ServerType.STDIO:
            self.connection = StdioMCPConnection(self.config)
        else:
            raise NotImplementedError(f"Server type {self.config.type} not yet implemented")
        
        # Set up reconnection handler
        self.connection.on_state_change(self._handle_state_change)
        
        # Connect
        success = await self.connection.connect()
        
        if success:
            self._reconnect_attempts = 0
        
        return success
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        if self.connection:
            await self.connection.disconnect()
            self.connection = None
    
    def _handle_state_change(self, state: ConnectionState):
        """Handle connection state changes.
        
        Args:
            state: New connection state
        """
        if state == ConnectionState.FAILED and self.config.retry_attempts > 0:
            # Start reconnection
            if not self._reconnect_task or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self):
        """Attempt to reconnect to MCP server."""
        while self._reconnect_attempts < self.config.retry_attempts:
            self._reconnect_attempts += 1
            logger.info(f"Reconnection attempt {self._reconnect_attempts}/{self.config.retry_attempts} for '{self.config.name}'")
            
            # Wait before reconnecting
            await asyncio.sleep(self.config.retry_delay * self._reconnect_attempts)
            
            try:
                if await self.connection.connect():
                    self._reconnect_attempts = 0
                    return
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
        
        logger.error(f"Failed to reconnect to '{self.config.name}' after {self.config.retry_attempts} attempts")
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.connection and self.connection.state == ConnectionState.CONNECTED
    
    @property
    def tools(self) -> List[MCPTool]:
        """Get available tools."""
        return self.connection.tools if self.connection else []
    
    @property
    def resources(self) -> List[MCPResource]:
        """Get available resources."""
        return self.connection.resources if self.connection else []
    
    @property
    def prompts(self) -> List[MCPPrompt]:
        """Get available prompts."""
        return self.connection.prompts if self.connection else []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        if not self.connection:
            raise MCPConnectionError(self.config.name, "Not connected")
        
        return await self.connection.call_tool(name, arguments)
    
    async def read_resource(self, uri: str) -> Any:
        """Read an MCP resource.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource content
        """
        if not self.connection:
            raise MCPConnectionError(self.config.name, "Not connected")
        
        return await self.connection.read_resource(uri)
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Get an MCP prompt.
        
        Args:
            name: Prompt name
            arguments: Prompt arguments
            
        Returns:
            Expanded prompt text
        """
        if not self.connection:
            raise MCPConnectionError(self.config.name, "Not connected")
        
        return await self.connection.get_prompt(name, arguments)


# Import os for environment variables
import os