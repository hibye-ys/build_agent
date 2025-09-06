"""Basic agent example demonstrating model switching and tool usage."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelProvider, ModelConfig, create_chat_model
from src.agents import create_react_agent_wrapper, ReactAgentConfig
from src.tools import get_tools_by_names
from src.config import get_model_config_for_example, get_validated_model_config


def main():
    """Run basic agent examples."""
    
    print("🤖 Multi-Model Agent Testing Framework")
    print("=" * 50)
    
    # Load model configurations for this example
    model_configs = get_model_config_for_example("basic_agent")
    print("\n📋 Using models from config:")
    print(f"  OpenAI: {model_configs.get('openai_model', 'default')}")
    print(f"  Anthropic: {model_configs.get('anthropic_model', 'default')}")
    print(f"  Google: {model_configs.get('google_model', 'default')}")
    
    # Example 1: OpenAI Agent
    print("\n📌 Example 1: OpenAI Agent")
    print("-" * 30)
    
    # Get validated model for OpenAI
    openai_model, is_original = get_validated_model_config("openai", model_configs.get('openai_model'))
    if not is_original:
        print(f"  ⚠️ Using alternative model: {openai_model}")
    
    openai_config = ReactAgentConfig(
        model_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name=openai_model,
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
        }, config={"configurable": {"thread_id": "example1"}})
        
        print("Response:", response["messages"][-1].content)
        
    except Exception as e:
        print(f"Error with OpenAI agent: {e}")
    
    # Example 2: Anthropic Agent  
    print(f"\n📌 Example 2: Anthropic Agent")
    print("-" * 30)
    
    # Get validated model for Anthropic
    anthropic_model, is_original = get_validated_model_config("anthropic", model_configs.get('anthropic_model'))
    if not is_original:
        print(f"  ⚠️ Using alternative model: {anthropic_model}")
    
    anthropic_config = ReactAgentConfig(
        model_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name=anthropic_model,
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
        }, config={"configurable": {"thread_id": "example2"}})
        
        print("Response:", response["messages"][-1].content)
        
    except Exception as e:
        print(f"Error with Anthropic agent: {e}")
    
    # Example 3: Google Agent
    print(f"\n📌 Example 3: Google Agent")
    print("-" * 30)
    
    # Get validated model for Google
    google_model, is_original = get_validated_model_config("google", model_configs.get('google_model'))
    if not is_original:
        print(f"  ⚠️ Using alternative model: {google_model}")
    
    google_config = ReactAgentConfig(
        model_config=ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_name=google_model,
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
        ("OpenAI", ModelProvider.OPENAI, openai_model),
        ("Anthropic", ModelProvider.ANTHROPIC, anthropic_model),  
        ("Google", ModelProvider.GOOGLE, google_model)
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