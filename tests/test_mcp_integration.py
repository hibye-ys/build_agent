"""
Tests for MCP integration module.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.mcp import (
    MCPManager,
    MCPConfig,
    ServerConfig,
    ServerType,
    MCPRegistry,
    MCPLangChainAdapter,
    MCPResourceManager,
    MCPException,
    MCPConnectionError,
    MCPServerNotFoundError
)
from src.mcp.client import MCPClient, MCPTool, MCPResource, ConnectionState


class TestMCPConfig:
    """Test MCP configuration management."""
    
    def test_server_config_creation(self):
        """Test creating server configuration."""
        config = ServerConfig(
            name="test_server",
            type=ServerType.STDIO,
            command="test_command",
            args=["arg1", "arg2"],
            timeout=30
        )
        
        assert config.name == "test_server"
        assert config.type == ServerType.STDIO
        assert config.command == "test_command"
        assert config.args == ["arg1", "arg2"]
        assert config.timeout == 30
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "type": "stdio",
            "command": "npx",
            "args": ["server"],
            "timeout": 45
        }
        
        config = ServerConfig.from_dict("test", data)
        assert config.name == "test"
        assert config.type == ServerType.STDIO
        assert config.command == "npx"
        assert config.timeout == 45
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = MCPConfig()
        
        # Add invalid server
        invalid_server = ServerConfig(
            name="invalid",
            type=ServerType.STDIO,
            command=None,  # Missing command
            timeout=-1  # Invalid timeout
        )
        config.add_server(invalid_server)
        
        warnings = config.validate()
        assert len(warnings) > 0
        assert any("missing command" in w for w in warnings)
        assert any("Invalid timeout" in w for w in warnings)
    
    def test_config_env_resolution(self):
        """Test environment variable resolution."""
        import os
        os.environ["TEST_TOKEN"] = "secret_token"
        
        data = {
            "type": "http",
            "url": "http://localhost",
            "auth": {
                "type": "bearer",
                "token": "${TEST_TOKEN}"
            }
        }
        
        config = ServerConfig.from_dict("test", data)
        assert config.auth.token == "secret_token"
        
        del os.environ["TEST_TOKEN"]


class TestMCPClient:
    """Test MCP client functionality."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization."""
        config = ServerConfig(
            name="test",
            type=ServerType.STDIO,
            command="echo",
            args=["test"]
        )
        
        client = MCPClient(config)
        assert client.config == config
        assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_connection_states(self):
        """Test connection state transitions."""
        config = ServerConfig(
            name="test",
            type=ServerType.STDIO,
            command="echo",
            args=["test"]
        )
        
        client = MCPClient(config)
        
        # Mock connection
        with patch.object(client, 'connection') as mock_conn:
            mock_conn.state = ConnectionState.DISCONNECTED
            assert not client.is_connected
            
            mock_conn.state = ConnectionState.CONNECTED
            assert client.is_connected
            
            mock_conn.state = ConnectionState.FAILED
            assert not client.is_connected


class TestMCPRegistry:
    """Test MCP registry functionality."""
    
    @pytest.mark.asyncio
    async def test_server_registration(self):
        """Test registering servers."""
        registry = MCPRegistry()
        
        config = ServerConfig(
            name="test_server",
            type=ServerType.STDIO,
            command="echo"
        )
        
        server_info = await registry.register_server(config)
        assert server_info.name == "test_server"
        assert server_info.config == config
        assert "test_server" in registry.list_servers()
    
    @pytest.mark.asyncio
    async def test_server_unregistration(self):
        """Test unregistering servers."""
        registry = MCPRegistry()
        
        config = ServerConfig(
            name="test_server",
            type=ServerType.STDIO,
            command="echo"
        )
        
        await registry.register_server(config)
        assert "test_server" in registry.list_servers()
        
        success = await registry.unregister_server("test_server")
        assert success
        assert "test_server" not in registry.list_servers()
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        registry = MCPRegistry()
        
        config = ServerConfig(
            name="test_server",
            type=ServerType.STDIO,
            command="echo"
        )
        
        server_info = await registry.register_server(config)
        
        # Mock client for health check
        mock_client = AsyncMock()
        mock_client.connection.health_check.return_value = True
        server_info.client = mock_client
        
        is_healthy = await registry.health_check("test_server")
        assert is_healthy
        assert server_info.status.value == "healthy"
    
    def test_server_selection_strategies(self):
        """Test different server selection strategies."""
        registry = MCPRegistry()
        
        # Add mock servers
        for i in range(3):
            server_info = MagicMock()
            server_info.status.value = "healthy"
            server_info.total_calls = i * 10
            server_info.success_rate = 1.0 - (i * 0.1)
            registry.servers[f"server_{i}"] = server_info
        
        # Mock healthy servers
        with patch.object(registry, 'get_healthy_servers') as mock_healthy:
            mock_healthy.return_value = ["server_0", "server_1", "server_2"]
            
            # Test round-robin (returns first)
            selected = registry.select_server(strategy="round_robin")
            assert selected == "server_0"
            
            # Test least loaded
            selected = registry.select_server(strategy="least_loaded")
            assert selected == "server_0"
            
            # Test best success rate
            selected = registry.select_server(strategy="best_success_rate")
            assert selected == "server_0"


class TestMCPLangChainAdapter:
    """Test LangChain adapter functionality."""
    
    def test_tool_conversion(self):
        """Test converting MCP tool to LangChain tool."""
        mcp_tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "integer"}
                },
                "required": ["param1"]
            }
        )
        
        mock_client = Mock()
        
        from src.mcp.adapter import MCPToolConverter
        langchain_tool = MCPToolConverter.convert_tool(mcp_tool, mock_client)
        
        assert langchain_tool.name == "test_tool"
        assert langchain_tool.description == "A test tool"
        assert hasattr(langchain_tool, 'args_schema')
    
    @pytest.mark.asyncio
    async def test_adapter_tool_creation(self):
        """Test adapter creating tools from client."""
        adapter = MCPLangChainAdapter()
        
        # Mock client
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.config.name = "test_server"
        mock_client.tools = [
            MCPTool(
                name="tool1",
                description="Tool 1",
                input_schema={"type": "object", "properties": {}}
            ),
            MCPTool(
                name="tool2",
                description="Tool 2",
                input_schema={"type": "object", "properties": {}}
            )
        ]
        
        tools = await adapter.create_tools(mock_client)
        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"
    
    @pytest.mark.asyncio
    async def test_tool_filtering(self):
        """Test tool filtering in adapter."""
        adapter = MCPLangChainAdapter()
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.config.name = "test_server"
        mock_client.tools = [
            MCPTool(name="tool1", description="", input_schema={}),
            MCPTool(name="tool2", description="", input_schema={}),
            MCPTool(name="tool3", description="", input_schema={})
        ]
        
        # Test filter_tools
        tools = await adapter.create_tools(
            mock_client,
            filter_tools=["tool1", "tool3"]
        )
        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool3"
        
        # Test exclude_tools
        tools = await adapter.create_tools(
            mock_client,
            exclude_tools=["tool2"]
        )
        assert len(tools) == 2
        assert all(t.name != "tool2" for t in tools)


class TestMCPResourceManager:
    """Test resource management functionality."""
    
    @pytest.mark.asyncio
    async def test_resource_discovery(self):
        """Test discovering resources."""
        registry = MCPRegistry()
        manager = MCPResourceManager(registry)
        
        # Mock client with resources
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.resources = [
            MCPResource(uri="file://test1", name="Test 1"),
            MCPResource(uri="file://test2", name="Test 2")
        ]
        
        with patch.object(registry, 'get_client') as mock_get:
            mock_get.return_value = mock_client
            registry.servers["test_server"] = Mock()
            
            discovered = await manager.discover_resources("test_server")
            assert "test_server" in discovered
            assert len(discovered["test_server"]) == 2
    
    @pytest.mark.asyncio
    async def test_resource_caching(self):
        """Test resource caching."""
        manager = MCPResourceManager(cache_enabled=True, cache_ttl=60)
        
        # Mock registry and client
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.read_resource.return_value = "cached_content"
        mock_client.resources = []
        
        mock_registry = Mock()
        mock_registry.get_client.return_value = mock_client
        mock_registry.get_healthy_servers.return_value = ["test_server"]
        mock_registry.get_server.return_value = Mock()
        
        manager.registry = mock_registry
        manager._resource_index["test://resource"] = ["test_server"]
        
        # First read - should call client
        content1 = await manager.read_resource("test://resource", use_cache=True)
        assert content1 == "cached_content"
        mock_client.read_resource.assert_called_once()
        
        # Second read - should use cache
        mock_client.read_resource.reset_mock()
        content2 = await manager.read_resource("test://resource", use_cache=True)
        assert content2 == "cached_content"
        mock_client.read_resource.assert_not_called()
    
    def test_resource_search(self):
        """Test searching resources."""
        manager = MCPResourceManager()
        
        # Add mock resources to search
        manager.list_resources = Mock(return_value=[
            {"uri": "file://config.yaml", "name": "Config File", "server": "s1"},
            {"uri": "file://data.json", "name": "Data File", "server": "s2"},
            {"uri": "file://config.ini", "name": "Settings", "server": "s3"}
        ])
        
        # Search for "config"
        results = manager.search_resources("config")
        assert len(results) == 2
        assert all("config" in r["uri"].lower() or "config" in r["name"].lower() for r in results)


class TestMCPManager:
    """Test main MCP manager."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager initialization."""
        manager = MCPManager()
        assert manager.config is not None
        assert manager.registry is not None
        assert manager.adapter is not None
        assert manager.resource_manager is not None
        assert not manager._initialized
        
        success = await manager.initialize()
        assert success
        assert manager._initialized
    
    @pytest.mark.asyncio
    async def test_manager_from_config(self):
        """Test creating manager from config file."""
        # Create temporary config
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "settings": {
                    "connection_timeout": 30
                },
                "servers": {
                    "test_server": {
                        "type": "stdio",
                        "command": "echo"
                    }
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = MCPManager.from_config(config_path)
            assert "test_server" in manager.config.servers
            assert manager.config.settings.connection_timeout == 30
        finally:
            Path(config_path).unlink()
    
    @pytest.mark.asyncio
    async def test_manager_tool_operations(self):
        """Test manager tool operations."""
        manager = MCPManager()
        await manager.initialize()
        
        # Mock a connected client
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.config.name = "test_server"
        mock_client.tools = [
            MCPTool(name="test_tool", description="Test", input_schema={})
        ]
        
        manager.registry.servers["test_server"] = Mock()
        manager.registry.get_client = Mock(return_value=mock_client)
        manager.registry.get_healthy_servers = Mock(return_value=["test_server"])
        
        # Get tools
        tools = await manager.get_langchain_tools()
        assert len(tools) > 0
        
        # List tool names
        tool_names = manager.list_tool_names()
        assert "test_tool" in tool_names
    
    @pytest.mark.asyncio
    async def test_manager_context_manager(self):
        """Test manager as context manager."""
        async with MCPManager() as manager:
            assert manager._initialized
        
        # After exiting, should be shutdown
        assert not manager._initialized


@pytest.mark.asyncio
async def test_integration_with_tool_registry():
    """Test integration with tool registry."""
    from src.tools.tool_registry import (
        set_mcp_manager,
        load_mcp_tools,
        list_mcp_tools,
        clear_mcp_tools
    )
    
    # Create mock manager
    mock_manager = AsyncMock()
    mock_manager.get_langchain_tools.return_value = [
        Mock(name="mcp_tool1"),
        Mock(name="mcp_tool2")
    ]
    
    # Set manager in registry
    set_mcp_manager(mock_manager)
    
    # Load MCP tools
    loaded = await load_mcp_tools()
    assert "mcp_tool1" in loaded
    assert "mcp_tool2" in loaded
    
    # List MCP tools
    mcp_tools = list_mcp_tools()
    assert "mcp_tool1" in mcp_tools
    assert "mcp_tool2" in mcp_tools
    
    # Clear MCP tools
    clear_mcp_tools()
    mcp_tools = list_mcp_tools()
    assert len(mcp_tools) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])