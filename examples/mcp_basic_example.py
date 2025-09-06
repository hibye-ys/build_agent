"""
Basic example of using MCP integration with LangChain agents.
"""

import asyncio
import os
from pathlib import Path

from src.mcp import MCPManager
from src.models.model_factory import ModelConfig, create_chat_model
from src.agents.custom_agent import create_custom_graph_agent
from src.tools.tool_registry import (
    set_mcp_manager,
    load_mcp_tools,
    list_mcp_tools,
    list_available_tools
)


async def basic_mcp_example():
    """Basic example of MCP integration."""
    
    print("🚀 MCP Basic Example")
    print("-" * 50)
    
    # 1. Create MCP manager from config
    config_path = Path("config/mcp_config.yaml")
    mcp_manager = MCPManager.from_config(config_path)
    
    # 2. Initialize and connect to servers
    print("\n📡 Initializing MCP Manager...")
    await mcp_manager.initialize()
    
    print("\n🔌 Connecting to MCP servers...")
    connection_results = await mcp_manager.connect()
    for server, connected in connection_results.items():
        status = "✅" if connected else "❌"
        print(f"  {status} {server}")
    
    # 3. List available MCP tools
    print("\n🔧 Available MCP tools:")
    mcp_tools = mcp_manager.list_tool_names()
    for tool in mcp_tools[:5]:  # Show first 5 tools
        print(f"  - {tool}")
    if len(mcp_tools) > 5:
        print(f"  ... and {len(mcp_tools) - 5} more")
    
    # 4. Get LangChain tools
    print("\n🔗 Converting to LangChain tools...")
    langchain_tools = await mcp_manager.get_langchain_tools()
    print(f"  Created {len(langchain_tools)} LangChain tools")
    
    # 5. Example: Call a tool directly
    if mcp_tools:
        tool_name = mcp_tools[0]
        print(f"\n📞 Calling tool '{tool_name}' directly...")
        try:
            # This is a demo - actual arguments depend on the tool
            result = await mcp_manager.call_tool(
                tool_name,
                arguments={}
            )
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # 6. List resources
    print("\n📚 Available resources:")
    resources = mcp_manager.list_resources()
    for resource in resources[:3]:  # Show first 3 resources
        print(f"  - {resource['name']} ({resource['uri']})")
    if len(resources) > 3:
        print(f"  ... and {len(resources) - 3} more")
    
    # 7. Get server statistics
    print("\n📊 Server statistics:")
    stats = mcp_manager.get_stats()
    for server_name, server_stats in stats['servers'].items():
        print(f"  {server_name}:")
        print(f"    Status: {server_stats['status']}")
        print(f"    Tools: {server_stats['tool_count']}")
        print(f"    Resources: {server_stats['resource_count']}")
    
    # 8. Shutdown
    print("\n🛑 Shutting down MCP Manager...")
    await mcp_manager.shutdown()
    print("✅ Done!")


async def mcp_with_agent_example():
    """Example of using MCP tools with a LangChain agent."""
    
    print("\n🤖 MCP with Agent Example")
    print("-" * 50)
    
    # 1. Setup MCP manager
    config_path = Path("config/mcp_config.yaml")
    mcp_manager = MCPManager.from_config(config_path)
    await mcp_manager.initialize()
    await mcp_manager.connect()
    
    # 2. Set MCP manager in tool registry
    set_mcp_manager(mcp_manager)
    
    # 3. Load MCP tools into registry
    print("\n📥 Loading MCP tools into registry...")
    loaded_tools = await load_mcp_tools()
    print(f"  Loaded {len(loaded_tools)} MCP tools")
    
    # 4. List all available tools (regular + MCP)
    all_tools = list_available_tools()
    print(f"\n🔧 Total available tools: {len(all_tools)}")
    print(f"  Regular tools: {len(all_tools) - len(loaded_tools)}")
    print(f"  MCP tools: {len(loaded_tools)}")
    
    # 5. Create agent with MCP tools
    print("\n🤖 Creating agent with MCP tools...")
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7
    )
    
    # Include both regular and MCP tools
    agent = create_custom_graph_agent(
        model_config=model_config,
        tools=["calculator", "weather"] + list_mcp_tools()[:3],  # Mix regular and MCP tools
        max_steps=5
    )
    
    # 6. Run agent with a task
    print("\n▶️ Running agent with task...")
    result = await agent.ainvoke({
        "messages": [
            ("human", "What's the weather like and calculate 25 * 4 for me?")
        ]
    })
    
    print("\n📝 Agent response:")
    for message in result["messages"]:
        print(f"  {message}")
    
    # 7. Cleanup
    await mcp_manager.shutdown()
    print("\n✅ Agent example complete!")


async def mcp_with_resources_example():
    """Example of working with MCP resources."""
    
    print("\n📦 MCP Resources Example")
    print("-" * 50)
    
    # 1. Setup MCP manager
    config_path = Path("config/mcp_config.yaml")
    mcp_manager = MCPManager.from_config(config_path)
    await mcp_manager.initialize()
    await mcp_manager.connect()
    
    # 2. Discover resources
    print("\n🔍 Discovering resources...")
    resources = mcp_manager.list_resources()
    
    if resources:
        # 3. Read a resource
        first_resource = resources[0]
        print(f"\n📖 Reading resource: {first_resource['name']}")
        try:
            content = await mcp_manager.read_resource(first_resource['uri'])
            print(f"  Content type: {type(content)}")
            print(f"  Content preview: {str(content)[:200]}...")
        except Exception as e:
            print(f"  Error reading resource: {e}")
        
        # 4. Search resources
        print("\n🔎 Searching for resources...")
        search_results = mcp_manager.resource_manager.search_resources("config")
        print(f"  Found {len(search_results)} resources matching 'config'")
        
        # 5. Cache statistics
        print("\n💾 Cache statistics:")
        cache_stats = mcp_manager.resource_manager.get_cache_stats()
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")
    else:
        print("  No resources available")
    
    # 6. Cleanup
    await mcp_manager.shutdown()
    print("\n✅ Resources example complete!")


async def main():
    """Run all examples."""
    
    # Check if config exists
    config_path = Path("config/mcp_config.yaml")
    if not config_path.exists():
        print("⚠️ Please create config/mcp_config.yaml first!")
        print("See config/mcp_config_example.yaml for an example.")
        return
    
    # Run examples
    print("=" * 60)
    print("MCP Integration Examples")
    print("=" * 60)
    
    try:
        # Basic example
        await basic_mcp_example()
        
        # Agent example
        await mcp_with_agent_example()
        
        # Resources example
        await mcp_with_resources_example()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())