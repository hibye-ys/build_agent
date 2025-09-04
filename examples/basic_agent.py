"""Basic agent example demonstrating model switching and tool usage."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelProvider, ModelConfig, create_chat_model
from src.agents import create_react_agent_wrapper, ReactAgentConfig
from src.tools import get_tools_by_names


def main():
    """Run basic agent examples."""
    
    print("🤖 Multi-Model Agent Testing Framework")
    print("=" * 50)
    
    # Example 1: OpenAI Agent
    print("\n📌 Example 1: OpenAI GPT-4 Agent")
    print("-" * 30)
    
    openai_config = ReactAgentConfig(
        model_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7
        ),
        tools=["calculator", "weather", "datetime"],
        system_prompt="You are a helpful assistant with access to tools.",
        memory=True
    )
    
    try:
        openai_agent = create_react_agent_wrapper(openai_config)
        
        # Test the agent
        response = openai_agent.invoke({
            "messages": [
                {"role": "user", "content": "What's 25 * 4? Also, what's the weather in Seoul?"}
            ]
        })
        
        print("Response:", response["messages"][-1].content)
        
    except Exception as e:
        print(f"Error with OpenAI agent: {e}")
    
    # Example 2: Anthropic Agent
    print("\n📌 Example 2: Anthropic Claude Agent")
    print("-" * 30)
    
    anthropic_config = ReactAgentConfig(
        model_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku",
            temperature=0.5
        ),
        tools=["calculator", "web_search"],
        system_prompt="You are Claude, a helpful AI assistant.",
        memory=True
    )
    
    try:
        anthropic_agent = create_react_agent_wrapper(anthropic_config)
        
        # Test the agent
        response = anthropic_agent.invoke({
            "messages": [
                {"role": "user", "content": "Calculate the square root of 144 and search for 'LangChain tutorial'"}
            ]
        })
        
        print("Response:", response["messages"][-1].content)
        
    except Exception as e:
        print(f"Error with Anthropic agent: {e}")
    
    # Example 3: Google Gemini Agent
    print("\n📌 Example 3: Google Gemini Agent")
    print("-" * 30)
    
    google_config = ReactAgentConfig(
        model_config=ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-1.5-flash",
            temperature=0.6
        ),
        tools=["datetime", "calculator"],
        system_prompt="You are a Google AI assistant powered by Gemini.",
        memory=False
    )
    
    try:
        google_agent = create_react_agent_wrapper(google_config)
        
        # Test the agent
        response = google_agent.invoke({
            "messages": [
                {"role": "user", "content": "What's the current date and time? Calculate 2^8."}
            ]
        })
        
        print("Response:", response["messages"][-1].content)
        
    except Exception as e:
        print(f"Error with Google agent: {e}")
    
    # Example 4: Model Comparison
    print("\n📌 Example 4: Model Comparison")
    print("-" * 30)
    
    test_query = "Explain quantum computing in one sentence."
    
    models_to_test = [
        ("OpenAI GPT-4", ModelProvider.OPENAI, "gpt-4"),
        ("Anthropic Claude", ModelProvider.ANTHROPIC, "claude-3-haiku"),
        ("Google Gemini", ModelProvider.GOOGLE, "gemini-1.5-flash")
    ]
    
    for name, provider, model_name in models_to_test:
        try:
            model = create_chat_model(ModelConfig(
                provider=provider,
                model_name=model_name,
                temperature=0.7
            ))
            
            response = model.invoke(test_query)
            print(f"\n{name}:")
            print(f"  {response.content}")
            
        except Exception as e:
            print(f"\n{name}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("✅ Examples completed!")


if __name__ == "__main__":
    # Check for API keys
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("⚠️ Warning: Missing API keys:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set these in your .env file.")
        print("Some examples may fail without proper API keys.\n")
    
    main()