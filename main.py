"""Main entry point for the Multi-Model Agent Testing Framework."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.models import ModelProvider, ModelConfig, get_available_models
from src.agents import create_simple_react_agent
from src.tools import list_available_tools


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════╗
║     🤖 Multi-Model Agent Testing Framework 🤖        ║
║                                                       ║
║  Support for: OpenAI, Anthropic, Google Models       ║
║  Built with: LangChain & LangGraph                   ║
╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check environment setup."""
    print("🔍 Checking environment...")
    
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google"
    }
    
    available_providers = []
    missing_providers = []
    
    for key, provider in api_keys.items():
        if os.getenv(key):
            available_providers.append(provider)
            print(f"  ✅ {provider} API key found")
        else:
            missing_providers.append(provider)
            print(f"  ❌ {provider} API key missing")
    
    return available_providers, missing_providers


def interactive_mode():
    """Run interactive agent mode."""
    print("\n🎮 Interactive Agent Mode")
    print("-" * 40)
    
    # Get available providers
    available_providers, _ = check_environment()
    
    if not available_providers:
        print("❌ No API keys configured. Please set up at least one provider.")
        return
    
    # Select provider
    print("\nAvailable providers:")
    for i, provider in enumerate(available_providers, 1):
        print(f"  {i}. {provider}")
    
    try:
        choice = input("\nSelect provider (number): ")
        provider_name = available_providers[int(choice) - 1].lower()
    except (ValueError, IndexError):
        print("Invalid selection. Using first available provider.")
        provider_name = available_providers[0].lower()
    
    # Select model
    provider_map = {
        "openai": ("openai", "gpt-4"),
        "anthropic": ("anthropic", "claude-3-haiku"),
        "google": ("google", "gemini-1.5-flash")
    }
    
    provider_key, default_model = provider_map.get(provider_name, ("openai", "gpt-4"))
    
    # Show available tools
    print("\n📦 Available tools:")
    tools = list_available_tools()
    for tool in tools:
        print(f"  - {tool}")
    
    # Create agent
    print(f"\n🤖 Creating agent with {provider_name}...")
    
    try:
        agent = create_simple_react_agent(
            model_provider=provider_key,
            model_name=default_model,
            tool_names=["calculator", "weather", "datetime", "web_search"],
            system_prompt="You are a helpful AI assistant with access to various tools.",
            memory=True
        )
        
        print("✅ Agent created successfully!")
        print("\n💬 Start chatting (type 'exit' to quit):")
        print("-" * 40)
        
        # Chat loop
        config = {"configurable": {"thread_id": "main"}}
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break
            
            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config
                )
                
                # Print the response
                if "messages" in response and response["messages"]:
                    ai_message = response["messages"][-1]
                    print(f"\n🤖: {ai_message.content}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")


def run_examples():
    """Run example scripts."""
    print("\n📚 Running Examples")
    print("-" * 40)
    
    examples_dir = Path("examples")
    example_files = [
        "basic_agent.py",
        "multi_agent_collaboration.py",
        "custom_workflow.py",
        "model_comparison.py"
    ]
    
    print("\nAvailable examples:")
    for i, example in enumerate(example_files, 1):
        print(f"  {i}. {example}")
    
    try:
        choice = input("\nSelect example to run (number): ")
        example_file = example_files[int(choice) - 1]
        
        example_path = examples_dir / example_file
        if example_path.exists():
            print(f"\n▶️ Running {example_file}...")
            print("-" * 40)
            exec(open(example_path).read())
        else:
            print(f"❌ Example file not found: {example_path}")
    
    except (ValueError, IndexError):
        print("Invalid selection.")
    except Exception as e:
        print(f"❌ Error running example: {e}")


def main():
    """Main entry point."""
    print_banner()
    
    # Check environment
    available_providers, missing_providers = check_environment()
    
    if missing_providers:
        print(f"\n💡 Tip: Add missing API keys to .env file for full functionality")
    
    # Main menu
    while True:
        print("\n📋 Main Menu")
        print("-" * 40)
        print("1. Interactive Agent Mode")
        print("2. Run Examples")
        print("3. Show Available Models")
        print("4. Show Available Tools")
        print("5. Exit")
        
        choice = input("\nSelect option (number): ")
        
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            run_examples()
        elif choice == "3":
            print("\n📊 Available Models:")
            models = get_available_models()
            for provider, model_list in models.items():
                print(f"\n{provider}:")
                for model in model_list:
                    print(f"  - {model}")
        elif choice == "4":
            print("\n🛠️ Available Tools:")
            tools = list_available_tools()
            for tool in tools:
                print(f"  - {tool}")
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid selection. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)