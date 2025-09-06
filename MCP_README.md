# MCP (Model Context Protocol) Integration Module

A flexible and reusable MCP integration module for LangChain/LangGraph agents, enabling seamless connection to MCP servers and tool discovery.

## 🌟 Features

- **Universal MCP Client**: Connect to any MCP server (STDIO, HTTP, WebSocket)
- **LangChain Integration**: Automatic conversion of MCP tools to LangChain format
- **Resource Management**: Efficient handling of MCP resources with caching
- **Health Monitoring**: Automatic health checks and server status tracking
- **Parallel Operations**: Optimized for concurrent server connections and tool calls
- **Dynamic Configuration**: Add/remove servers at runtime
- **Error Recovery**: Automatic reconnection with exponential backoff
- **Tool Registry Integration**: Seamless integration with existing LangChain tools

## 📦 Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install MCP servers (via npm):
```bash
# Filesystem server
npm install -g @modelcontextprotocol/server-filesystem

# Browser automation server
npm install -g @modelcontextprotocol/server-playwright

# GitHub server
npm install -g @modelcontextprotocol/server-github
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from src.mcp import MCPManager

async def main():
    # Create manager from config
    mcp = MCPManager.from_config("config/mcp_config.yaml")
    
    # Initialize and connect
    await mcp.initialize()
    await mcp.connect()
    
    # Get LangChain tools
    tools = await mcp.get_langchain_tools()
    
    # Use tools in your agent
    for tool in tools:
        print(f"Tool: {tool.name} - {tool.description}")
    
    # Cleanup
    await mcp.shutdown()

asyncio.run(main())
```

### Integration with LangChain Agent

```python
from src.mcp import MCPManager
from src.agents.custom_agent import create_custom_graph_agent
from src.tools.tool_registry import set_mcp_manager, load_mcp_tools

async def create_mcp_agent():
    # Setup MCP
    mcp = MCPManager.from_config("config/mcp_config.yaml")
    await mcp.initialize()
    await mcp.connect()
    
    # Integrate with tool registry
    set_mcp_manager(mcp)
    await load_mcp_tools()
    
    # Create agent with MCP tools
    agent = create_custom_graph_agent(
        model_config=model_config,
        tools=["calculator", "weather"] + list_mcp_tools(),
        max_steps=10
    )
    
    return agent, mcp
```

## 📁 Module Structure

```
src/mcp/
├── __init__.py          # Public API exports
├── client.py            # MCP client implementation
├── adapter.py           # LangChain adapter
├── config.py            # Configuration management
├── registry.py          # Server registry
├── resources.py         # Resource management
├── manager.py           # Main manager interface
└── exceptions.py        # Custom exceptions
```

## ⚙️ Configuration

### YAML Configuration

Create `config/mcp_config.yaml`:

```yaml
settings:
  connection_timeout: 30
  retry_attempts: 3
  lazy_loading: true
  cache_enabled: true
  cache_ttl: 3600

servers:
  filesystem_mcp:
    type: stdio
    command: npx
    args:
      - "@modelcontextprotocol/server-filesystem"
      - "/shared/directory"
    timeout: 30
    
  browser_mcp:
    type: stdio
    command: npx
    args:
      - "@modelcontextprotocol/server-playwright"
    env:
      BROWSER_TYPE: chromium
```

### Dynamic Configuration

```python
from src.mcp import MCPManager, ServerConfig, ServerType

# Create manager
mcp = MCPManager()

# Add server dynamically
config = ServerConfig(
    name="my_server",
    type=ServerType.STDIO,
    command="npx",
    args=["@modelcontextprotocol/server-custom"],
    timeout=30
)

await mcp.add_server(config, connect=True)
```

## 🔧 Advanced Features

### Parallel Operations

```python
# Connect to multiple servers in parallel
results = await mcp.connect()  # Connects all servers concurrently

# Parallel tool calls
tasks = [
    mcp.call_tool("tool1", {"arg": "value1"}),
    mcp.call_tool("tool2", {"arg": "value2"}),
    mcp.call_tool("tool3", {"arg": "value3"})
]
results = await asyncio.gather(*tasks)
```

### Resource Management

```python
# List resources
resources = mcp.list_resources()

# Read resource with caching
content = await mcp.read_resource(
    uri="file://path/to/resource",
    use_cache=True
)

# Search resources
results = mcp.resource_manager.search_resources("config")

# Prefetch resources
await mcp.resource_manager.prefetch_resources([
    "file://resource1",
    "file://resource2"
])
```

### Health Monitoring

```python
# Enable health monitoring
async with MCPManager.from_config("config.yaml") as mcp:
    # Health checks run automatically
    
    # Manual health check
    health_status = await mcp.health_check()
    
    # Get server statistics
    stats = mcp.get_stats()
    print(f"Connected servers: {stats['connected_servers']}")
    print(f"Total tools: {stats['total_tools']}")
```

### Error Handling

```python
from src.mcp.exceptions import (
    MCPConnectionError,
    MCPServerNotFoundError,
    MCPToolConversionError
)

try:
    await mcp.connect("server_name")
except MCPConnectionError as e:
    print(f"Connection failed: {e}")
    # Automatic retry with exponential backoff

try:
    result = await mcp.call_tool("tool_name", {})
except MCPServerNotFoundError as e:
    print(f"Server not found: {e}")
    print(f"Available servers: {e.available_servers}")
```

## 🏗️ Architecture

### Component Overview

1. **MCPManager**: Main interface for all MCP operations
2. **MCPClient**: Handles individual server connections
3. **MCPRegistry**: Manages multiple servers and health monitoring
4. **MCPLangChainAdapter**: Converts MCP tools to LangChain format
5. **MCPResourceManager**: Handles resource operations with caching
6. **MCPConfig**: Configuration management and validation

### Connection Types

- **STDIO**: Process-based communication (most common)
- **HTTP**: REST API endpoints (planned)
- **WebSocket**: Real-time bidirectional communication (planned)
- **SSE**: Server-sent events (planned)

### Tool Conversion

MCP tools are automatically converted to LangChain `BaseTool` instances:

1. **Schema Mapping**: JSON Schema → Pydantic models
2. **Async Support**: All MCP tools are async-compatible
3. **Error Handling**: Graceful error handling with fallbacks
4. **Lazy Loading**: Tools loaded on-demand for performance

## 📊 Performance Considerations

- **Connection Pooling**: Reuse connections across operations
- **Parallel Execution**: Batch operations when possible
- **Resource Caching**: Configurable TTL-based caching
- **Lazy Loading**: Load tools only when needed
- **Health Monitoring**: Automatic failover to healthy servers

## 🧪 Testing

Run the examples:

```bash
# Basic examples
python examples/mcp_basic_example.py

# Advanced examples
python examples/mcp_advanced_example.py
```

## 🤝 Integration Examples

### With Tool Registry

```python
from src.tools.tool_registry import set_mcp_manager, load_mcp_tools

# Set MCP manager
set_mcp_manager(mcp_manager)

# Load MCP tools
await load_mcp_tools(
    server_name="filesystem_mcp",
    filter_tools=["read_file", "write_file"]
)

# Use in agent
tools = get_tools_by_names(["calculator", "read_file"])
```

### With LangGraph Workflow

```python
from langgraph.graph import StateGraph

# Create workflow with MCP tools
workflow = StateGraph(AgentState)

# Add MCP-powered node
async def mcp_node(state):
    tools = await mcp.get_langchain_tools()
    # Process with tools
    return state

workflow.add_node("mcp_analysis", mcp_node)
```

## 🔍 Troubleshooting

### Common Issues

1. **Server Not Starting**
   - Check if MCP server is installed: `npm list -g @modelcontextprotocol/server-name`
   - Verify command path and arguments in config

2. **Connection Timeout**
   - Increase `connection_timeout` in settings
   - Check server logs for errors

3. **Tool Conversion Errors**
   - Ensure tool schemas are valid JSON Schema
   - Check for unsupported types in schema

4. **Resource Not Found**
   - Verify resource URI format
   - Check server has access to resource

## 📚 References

- [Model Context Protocol Specification](https://modelcontextprotocol.io)
- [LangChain Documentation](https://docs.langchain.com)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## 🔮 Future Enhancements

- [ ] HTTP/WebSocket connection support
- [ ] Advanced load balancing strategies
- [ ] Tool versioning and compatibility checks
- [ ] Metrics and observability integration
- [ ] Plugin system for custom adapters
- [ ] GUI for server management
- [ ] Automated server discovery