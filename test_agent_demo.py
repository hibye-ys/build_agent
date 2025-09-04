#!/usr/bin/env python3
"""
Test Agent Demo - Demonstration of the Multi-Model Agent Testing Framework
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project path
project_path = Path(".conductor/hibye-ys-dublin")
sys.path.insert(0, str(project_path))

# Load environment
load_dotenv()

from src.models import ModelProvider, ModelConfig
from src.agents import create_simple_react_agent
from src.tools import list_available_tools

def check_api_keys():
    """Check which API keys are available."""
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic", 
        "GOOGLE_API_KEY": "Google"
    }
    
    available = []
    for key, provider in api_keys.items():
        if os.getenv(key):
            available.append(provider.lower())
            print(f"✅ {provider} API key found")
        else:
            print(f"❌ {provider} API key missing")
    
    return available

def create_test_agent(provider="openai", model="gpt-4o-mini"):
    """Create a test agent with standard configuration."""
    
    system_prompt = """You are an intelligent AI assistant with access to various tools. 
    You should:
    1. Use tools when they can help provide accurate information
    2. Be conversational and helpful
    3. Show your reasoning process
    4. Handle complex multi-step tasks efficiently
    """
    
    try:
        agent = create_simple_react_agent(
            model_provider=provider,
            model_name=model,
            tool_names=["calculator", "weather", "datetime", "web_search"],
            system_prompt=system_prompt,
            memory=True
        )
        print(f"✅ Test agent created: {provider}:{model}")
        return agent
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None

def run_test_scenarios(agent, agent_name="Test Agent"):
    """Run a series of test scenarios."""
    
    if not agent:
        print("❌ No agent available for testing")
        return
    
    test_scenarios = [
        {
            "name": "Basic Math",
            "query": "What's 15 * 23 + 47?",
            "expected_tools": ["calculator"]
        },
        {
            "name": "Weather Query", 
            "query": "What's the weather like in Seoul today?",
            "expected_tools": ["weather"]
        },
        {
            "name": "Time Information",
            "query": "What's the current date and time?",
            "expected_tools": ["datetime"]
        },
        {
            "name": "Complex Multi-Tool",
            "query": "What's the current time? Also calculate 25% of 2840, and tell me the weather in Tokyo.",
            "expected_tools": ["datetime", "calculator", "weather"]
        },
        {
            "name": "Web Search",
            "query": "Search for recent news about artificial intelligence developments",
            "expected_tools": ["web_search"]
        },
        {
            "name": "Problem Solving",
            "query": "I have a rectangular garden that's 12.5 meters long and 8.3 meters wide. What's the area? If I want to plant flowers that need 0.5 square meters each, how many can I plant?",
            "expected_tools": ["calculator"]
        }
    ]
    
    print(f"\n🧪 Running Test Scenarios for {agent_name}")
    print("=" * 60)
    
    # Use consistent thread for memory testing
    config = {"configurable": {"thread_id": "demo_conversation"}}
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Test {i}: {scenario['name']}")
        print(f"📝 Query: {scenario['query']}")
        print("-" * 40)
        
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": scenario['query']}]
            }, config=config)
            
            # Extract response
            if "messages" in response and response["messages"]:
                ai_message = response["messages"][-1]
                content = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)
                print(f"✅ Response:\n{content}")
            else:
                print("❌ No response content found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)

def run_memory_test(agent, agent_name="Test Agent"):
    """Test conversation memory capabilities."""
    
    if not agent:
        return
        
    print(f"\n🧠 Memory Test for {agent_name}")
    print("=" * 50)
    
    memory_conversation = [
        "Hi, my name is Sarah and I work as a software engineer in Seoul.",
        "What's 150 divided by 6?", 
        "Thanks! What did I tell you my profession was?",
        "Correct! And what city did I mention I work in?",
        "Perfect memory! Now, what's the weather like in my city?"
    ]
    
    # Use dedicated thread for memory testing
    memory_config = {"configurable": {"thread_id": "memory_test_conversation"}}
    
    for i, message in enumerate(memory_conversation, 1):
        print(f"\n💬 Turn {i}: {message}")
        
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": message}]
            }, config=memory_config)
            
            if "messages" in response and response["messages"]:
                ai_message = response["messages"][-1]
                content = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)
                print(f"🤖 Response: {content}")
            else:
                print("❌ No response content found")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demonstration function."""
    
    print("🤖 Multi-Model Agent Testing Framework Demo")
    print("=" * 60)
    
    # Check available providers
    available_providers = check_api_keys()
    
    if not available_providers:
        print("\n❌ No API keys configured!")
        print("Please set up at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
        return
    
    print(f"\n🎯 Available providers: {', '.join(available_providers)}")
    
    # Show available tools
    print(f"\n🛠️ Available tools: {', '.join(list_available_tools())}")
    
    # Test with first available provider
    provider = available_providers[0]
    
    # Model selection
    model_map = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku",
        "google": "gemini-1.5-flash"
    }
    
    model = model_map.get(provider, "gpt-4o-mini")
    
    # Create test agent
    print(f"\n🚀 Creating test agent: {provider}:{model}")
    test_agent = create_test_agent(provider, model)
    
    if test_agent:
        # Run comprehensive tests
        run_test_scenarios(test_agent, f"{provider}:{model}")
        run_memory_test(test_agent, f"{provider}:{model}")
        
        print(f"\n🎉 Demo completed successfully with {provider}:{model}!")
        print("\n💡 Tips:")
        print("  • Check the Jupyter notebook for interactive testing")
        print("  • Modify this script to test different scenarios") 
        print("  • Add your own custom tools and workflows")
        
    else:
        print("\n❌ Could not create test agent. Check your API keys and configuration.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")