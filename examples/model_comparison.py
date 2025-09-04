"""Model comparison example for testing different providers."""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelProvider, ModelConfig, create_chat_model, get_available_models


def compare_models(
    prompt: str,
    models_to_test: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare responses from different models.
    
    Args:
        prompt: Test prompt
        models_to_test: List of model configurations
        
    Returns:
        Comparison results
    """
    results = {}
    
    for model_info in models_to_test:
        provider = model_info["provider"]
        model_name = model_info["model"]
        
        try:
            # Create model
            config = ModelConfig(
                provider=provider,
                model_name=model_name,
                temperature=0.7,
                max_tokens=150
            )
            model = create_chat_model(config)
            
            # Time the response
            start_time = time.time()
            response = model.invoke(prompt)
            elapsed_time = time.time() - start_time
            
            # Store results
            results[f"{provider.value}:{model_name}"] = {
                "response": response.content,
                "time": elapsed_time,
                "tokens": len(response.content.split()),
                "success": True
            }
            
        except Exception as e:
            results[f"{provider.value}:{model_name}"] = {
                "response": None,
                "error": str(e),
                "success": False
            }
    
    return results


def main():
    """Run model comparison examples."""
    
    print("🔬 Model Comparison Testing")
    print("=" * 50)
    
    # Show available models
    print("\n📋 Available Models:")
    print("-" * 30)
    
    available = get_available_models()
    for provider, models in available.items():
        print(f"\n{provider.upper()}:")
        for model in models:
            print(f"  - {model}")
    
    # Test prompts
    test_prompts = [
        {
            "name": "Creative Writing",
            "prompt": "Write a haiku about artificial intelligence."
        },
        {
            "name": "Logical Reasoning",
            "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to calculate the factorial of a number using recursion."
        },
        {
            "name": "Summarization",
            "prompt": "Summarize the importance of machine learning in modern technology in 2 sentences."
        }
    ]
    
    # Models to test (configure based on available API keys)
    models_to_test = []
    
    if os.getenv("OPENAI_API_KEY"):
        models_to_test.extend([
            {"provider": ModelProvider.OPENAI, "model": "gpt-4"},
            {"provider": ModelProvider.OPENAI, "model": "gpt-3.5-turbo"}
        ])
    
    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_test.extend([
            {"provider": ModelProvider.ANTHROPIC, "model": "claude-3-haiku"},
            {"provider": ModelProvider.ANTHROPIC, "model": "claude-3-sonnet"}
        ])
    
    if os.getenv("GOOGLE_API_KEY"):
        models_to_test.extend([
            {"provider": ModelProvider.GOOGLE, "model": "gemini-1.5-flash"},
            {"provider": ModelProvider.GOOGLE, "model": "gemini-pro"}
        ])
    
    if not models_to_test:
        print("\n⚠️ No API keys found. Please set at least one:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GOOGLE_API_KEY")
        return
    
    # Run comparisons
    for test_case in test_prompts:
        print(f"\n🎯 Test: {test_case['name']}")
        print("=" * 50)
        print(f"Prompt: {test_case['prompt']}")
        print("-" * 50)
        
        results = compare_models(test_case["prompt"], models_to_test)
        
        # Display results
        for model_id, result in results.items():
            print(f"\n📌 {model_id}")
            
            if result["success"]:
                print(f"⏱️ Response time: {result['time']:.2f}s")
                print(f"📝 Word count: {result['tokens']}")
                print(f"Response:")
                print(f"  {result['response']}")
            else:
                print(f"❌ Error: {result['error']}")
    
    # Performance summary
    print("\n" + "=" * 50)
    print("📊 Performance Summary")
    print("-" * 30)
    
    # Calculate average response times
    model_times = {}
    for test_case in test_prompts:
        results = compare_models(test_case["prompt"], models_to_test)
        for model_id, result in results.items():
            if result["success"]:
                if model_id not in model_times:
                    model_times[model_id] = []
                model_times[model_id].append(result["time"])
    
    for model_id, times in model_times.items():
        avg_time = sum(times) / len(times)
        print(f"{model_id}: {avg_time:.2f}s average")
    
    print("\n✅ Model comparison completed!")


if __name__ == "__main__":
    main()