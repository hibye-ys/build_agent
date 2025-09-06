"""
Advanced examples of MCP integration with LangChain/LangGraph.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START

from src.mcp import MCPManager, MCPConfig, ServerConfig, ServerType
from src.models.model_factory import ModelConfig, create_chat_model
from src.agents.custom_agent import AgentState
from src.tools.tool_registry import set_mcp_manager, load_mcp_tools


async def parallel_mcp_operations():
    """Example of parallel MCP operations for performance."""
    
    print("\n⚡ Parallel MCP Operations Example")
    print("-" * 50)
    
    # Setup MCP manager
    config_path = Path("config/mcp_config.yaml")
    mcp_manager = MCPManager.from_config(config_path)
    await mcp_manager.initialize()
    
    # Connect to all servers in parallel
    print("\n🔌 Connecting to servers in parallel...")
    start_time = time.time()
    connection_results = await mcp_manager.connect()
    connect_time = time.time() - start_time
    print(f"  Connected to {sum(connection_results.values())} servers in {connect_time:.2f}s")
    
    # Get all server names
    servers = mcp_manager.registry.get_healthy_servers()
    
    if len(servers) >= 2:
        # Parallel tool discovery
        print("\n🔧 Discovering tools from multiple servers in parallel...")
        start_time = time.time()
        
        tasks = [
            mcp_manager.get_langchain_tools(server_name=server)
            for server in servers
        ]
        
        results = await asyncio.gather(*tasks)
        discovery_time = time.time() - start_time
        
        total_tools = sum(len(tools) for tools in results)
        print(f"  Discovered {total_tools} tools in {discovery_time:.2f}s")
        
        # Parallel tool calls
        print("\n📞 Making parallel tool calls...")
        if total_tools > 0:
            # Get first available tool from each server
            tool_calls = []
            for server, tools in zip(servers, results):
                if tools:
                    tool_calls.append({
                        'server': server,
                        'tool': tools[0].name,
                        'args': {}
                    })
            
            if tool_calls:
                start_time = time.time()
                
                call_tasks = [
                    mcp_manager.call_tool(
                        tool_name=call['tool'],
                        arguments=call['args'],
                        server_name=call['server']
                    )
                    for call in tool_calls[:3]  # Limit to 3 parallel calls
                ]
                
                call_results = await asyncio.gather(*call_tasks, return_exceptions=True)
                call_time = time.time() - start_time
                
                print(f"  Made {len(call_tasks)} parallel calls in {call_time:.2f}s")
                for call, result in zip(tool_calls[:3], call_results):
                    if isinstance(result, Exception):
                        print(f"    {call['tool']}: Error - {result}")
                    else:
                        print(f"    {call['tool']}: Success")
    
    # Parallel resource reading
    print("\n📚 Reading resources in parallel...")
    resources = mcp_manager.list_resources()
    if resources:
        resource_uris = [r['uri'] for r in resources[:5]]  # Read first 5
        
        start_time = time.time()
        resource_contents = await mcp_manager.resource_manager.read_resources(
            uris=resource_uris,
            parallel=True,
            use_cache=True
        )
        read_time = time.time() - start_time
        
        successful_reads = sum(1 for v in resource_contents.values() if v is not None)
        print(f"  Read {successful_reads}/{len(resource_uris)} resources in {read_time:.2f}s")
    
    # Cleanup
    await mcp_manager.shutdown()
    print("\n✅ Parallel operations complete!")


async def dynamic_mcp_configuration():
    """Example of dynamic MCP server configuration."""
    
    print("\n🔧 Dynamic MCP Configuration Example")
    print("-" * 50)
    
    # Create manager without initial config
    mcp_manager = MCPManager()
    await mcp_manager.initialize()
    
    # Dynamically add servers
    print("\n➕ Adding servers dynamically...")
    
    # Add a filesystem MCP server
    filesystem_config = ServerConfig(
        name="dynamic_filesystem",
        type=ServerType.STDIO,
        command="npx",
        args=["@modelcontextprotocol/server-filesystem", "/tmp"],
        timeout=30
    )
    
    await mcp_manager.add_server(filesystem_config, connect=True)
    print(f"  Added and connected to 'dynamic_filesystem'")
    
    # Add another server (example)
    if Path("config/dynamic_server.json").exists():
        # Load from JSON
        import json
        with open("config/dynamic_server.json", "r") as f:
            server_data = json.load(f)
        
        dynamic_config = ServerConfig.from_dict("dynamic_custom", server_data)
        await mcp_manager.add_server(dynamic_config, connect=True)
        print(f"  Added and connected to 'dynamic_custom'")
    
    # List current servers
    print("\n📋 Current servers:")
    for server_name in mcp_manager.registry.list_servers():
        info = mcp_manager.get_server_info(server_name)
        print(f"  - {server_name}: {info.status.value}")
    
    # Use the dynamically added servers
    print("\n🔧 Tools from dynamic servers:")
    tools = await mcp_manager.get_langchain_tools()
    print(f"  Total tools: {len(tools)}")
    
    # Remove a server
    print("\n➖ Removing a server...")
    await mcp_manager.remove_server("dynamic_filesystem")
    print("  Removed 'dynamic_filesystem'")
    
    # Cleanup
    await mcp_manager.shutdown()
    print("\n✅ Dynamic configuration complete!")


async def mcp_with_langgraph_workflow():
    """Example of MCP integration in a LangGraph workflow."""
    
    print("\n🔄 MCP with LangGraph Workflow Example")
    print("-" * 50)
    
    # Setup MCP manager
    config_path = Path("config/mcp_config.yaml")
    mcp_manager = MCPManager.from_config(config_path)
    await mcp_manager.initialize()
    await mcp_manager.connect()
    
    # Get MCP tools
    mcp_tools = await mcp_manager.get_langchain_tools()
    
    if not mcp_tools:
        print("  No MCP tools available, skipping workflow example")
        await mcp_manager.shutdown()
        return
    
    # Create model
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7
    )
    model = create_chat_model(model_config)
    
    # Define workflow nodes
    async def analyze_with_mcp(state: AgentState) -> Dict[str, Any]:
        """Analyze using MCP tools."""
        messages = state["messages"]
        
        # Use first available MCP tool
        tool = mcp_tools[0]
        print(f"  Using MCP tool: {tool.name}")
        
        try:
            result = await tool._arun()
            return {
                "messages": messages + [AIMessage(content=f"MCP Analysis: {result}")]
            }
        except Exception as e:
            return {
                "messages": messages + [AIMessage(content=f"MCP Error: {e}")]
            }
    
    async def process_with_model(state: AgentState) -> Dict[str, Any]:
        """Process with language model."""
        messages = state["messages"]
        
        # Bind MCP tools to model
        model_with_tools = model.bind_tools(mcp_tools[:3])  # Use first 3 tools
        
        response = await model_with_tools.ainvoke(messages)
        return {"messages": messages + [response]}
    
    async def check_resources(state: AgentState) -> Dict[str, Any]:
        """Check MCP resources."""
        messages = state["messages"]
        
        resources = mcp_manager.list_resources()
        resource_summary = f"Found {len(resources)} MCP resources available"
        
        return {
            "messages": messages + [AIMessage(content=resource_summary)]
        }
    
    # Build workflow
    print("\n🏗️ Building LangGraph workflow with MCP...")
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_with_mcp)
    workflow.add_node("process", process_with_model)
    workflow.add_node("check_resources", check_resources)
    
    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "process")
    workflow.add_edge("process", "check_resources")
    workflow.add_edge("check_resources", END)
    
    # Compile workflow
    app = workflow.compile()
    
    # Run workflow
    print("\n▶️ Running workflow...")
    initial_state = {
        "messages": [HumanMessage(content="Analyze the current system state")]
    }
    
    result = await app.ainvoke(initial_state)
    
    print("\n📝 Workflow results:")
    for i, message in enumerate(result["messages"]):
        print(f"  Step {i + 1}: {message.content[:100]}...")
    
    # Cleanup
    await mcp_manager.shutdown()
    print("\n✅ Workflow example complete!")


async def mcp_health_monitoring():
    """Example of MCP server health monitoring."""
    
    print("\n🏥 MCP Health Monitoring Example")
    print("-" * 50)
    
    # Setup MCP manager with health monitoring
    config_path = Path("config/mcp_config.yaml")
    mcp_manager = MCPManager.from_config(config_path)
    await mcp_manager.initialize()
    await mcp_manager.connect()
    
    # Start health monitoring
    print("\n🔍 Starting health monitoring...")
    
    # Simulate monitoring for a short period
    monitoring_duration = 10  # seconds
    check_interval = 2  # seconds
    
    start_time = time.time()
    check_count = 0
    
    while time.time() - start_time < monitoring_duration:
        check_count += 1
        print(f"\n  Health check #{check_count}:")
        
        # Perform health check
        health_results = await mcp_manager.health_check()
        
        for server, is_healthy in health_results.items():
            status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
            server_info = mcp_manager.get_server_info(server)
            
            print(f"    {server}: {status}")
            if server_info:
                print(f"      - Total calls: {server_info.total_calls}")
                print(f"      - Success rate: {server_info.success_rate:.1%}")
                print(f"      - Error count: {server_info.error_count}")
        
        # Wait before next check
        await asyncio.sleep(check_interval)
    
    # Final statistics
    print("\n📊 Final Statistics:")
    stats = mcp_manager.get_stats()
    
    for server_name, server_stats in stats['servers'].items():
        print(f"  {server_name}:")
        print(f"    - Status: {server_stats['status']}")
        print(f"    - Total calls: {server_stats['total_calls']}")
        print(f"    - Success rate: {server_stats['success_rate']:.1%}")
    
    # Cleanup
    await mcp_manager.shutdown()
    print("\n✅ Health monitoring complete!")


async def mcp_error_handling():
    """Example of MCP error handling and recovery."""
    
    print("\n🛡️ MCP Error Handling Example")
    print("-" * 50)
    
    # Create manager
    mcp_manager = MCPManager()
    await mcp_manager.initialize()
    
    # Try to connect to a non-existent server
    print("\n❌ Testing connection failure...")
    fake_config = ServerConfig(
        name="fake_server",
        type=ServerType.STDIO,
        command="non_existent_command",
        args=["--fake"],
        retry_attempts=2,
        retry_delay=1.0
    )
    
    try:
        await mcp_manager.add_server(fake_config, connect=True)
    except Exception as e:
        print(f"  Expected error: {e}")
    
    # Add a real server
    if Path("config/mcp_config.yaml").exists():
        real_config = MCPConfig(Path("config/mcp_config.yaml"))
        if real_config.servers:
            first_server = list(real_config.servers.values())[0]
            await mcp_manager.add_server(first_server, connect=True)
            
            # Test tool error handling
            print("\n🔧 Testing tool error handling...")
            
            try:
                # Call tool with invalid arguments
                result = await mcp_manager.call_tool(
                    tool_name="non_existent_tool",
                    arguments={"invalid": "args"}
                )
            except Exception as e:
                print(f"  Expected error: {type(e).__name__}: {e}")
            
            # Test resource error handling
            print("\n📚 Testing resource error handling...")
            
            try:
                # Read non-existent resource
                content = await mcp_manager.read_resource(
                    uri="invalid://resource",
                    use_cache=False
                )
            except Exception as e:
                print(f"  Expected error: {type(e).__name__}: {e}")
    
    # Test recovery after errors
    print("\n🔄 Testing recovery...")
    health_check = await mcp_manager.health_check()
    healthy_count = sum(1 for healthy in health_check.values() if healthy)
    print(f"  Healthy servers after errors: {healthy_count}/{len(health_check)}")
    
    # Cleanup
    await mcp_manager.shutdown()
    print("\n✅ Error handling example complete!")


async def main():
    """Run advanced examples."""
    
    print("=" * 60)
    print("Advanced MCP Integration Examples")
    print("=" * 60)
    
    try:
        # Parallel operations
        await parallel_mcp_operations()
        
        # Dynamic configuration
        await dynamic_mcp_configuration()
        
        # LangGraph workflow
        await mcp_with_langgraph_workflow()
        
        # Health monitoring
        await mcp_health_monitoring()
        
        # Error handling
        await mcp_error_handling()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())