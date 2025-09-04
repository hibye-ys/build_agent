"""Multi-agent collaboration example."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelProvider, ModelConfig
from src.agents.multi_agent import AgentConfig, create_supervisor_agent


def main():
    """Run multi-agent collaboration example."""
    
    print("🤝 Multi-Agent Collaboration Example")
    print("=" * 50)
    
    # Define specialized agents
    research_agent_config = AgentConfig(
        name="research_agent",
        model_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7
        ),
        tools=["web_search", "calculator"],
        prompt="You are a research specialist. Your job is to find and analyze information.",
        description="Handles research tasks, data gathering, and information analysis"
    )
    
    writer_agent_config = AgentConfig(
        name="writer_agent",
        model_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.8
        ),
        tools=["datetime"],
        prompt="You are a professional writer. Create clear, engaging content based on provided information.",
        description="Handles content creation, writing, and documentation tasks"
    )
    
    analyst_agent_config = AgentConfig(
        name="analyst_agent",
        model_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.5
        ),
        tools=["calculator"],
        prompt="You are a data analyst. Analyze numerical data and provide insights.",
        description="Handles data analysis, calculations, and statistical insights"
    )
    
    # Create supervisor configuration
    supervisor_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.3
    )
    
    # Create the multi-agent system
    agents = {
        "research_agent": research_agent_config,
        "writer_agent": writer_agent_config,
        "analyst_agent": analyst_agent_config
    }
    
    try:
        multi_agent_system = create_supervisor_agent(
            supervisor_config=supervisor_config,
            agents=agents,
            supervisor_prompt="""You are a project supervisor managing three specialized agents:
- research_agent: For research and information gathering
- writer_agent: For content creation and documentation
- analyst_agent: For data analysis and calculations

Delegate tasks to the appropriate agent based on the requirement.
Ensure efficient task distribution and coordination."""
        )
        
        # Test scenarios
        test_scenarios = [
            "Research the latest developments in AI and write a brief summary.",
            "Calculate the compound interest on $10,000 at 5% annually for 10 years, then explain the result.",
            "Find information about renewable energy and create a short report."
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario}")
            print("-" * 40)
            
            try:
                response = multi_agent_system.invoke({
                    "messages": [
                        {"role": "user", "content": scenario}
                    ]
                })
                
                # Print the final response
                if "messages" in response and response["messages"]:
                    final_message = response["messages"][-1]
                    print(f"Response: {final_message.content}")
                
            except Exception as e:
                print(f"Error in scenario {i}: {e}")
        
    except Exception as e:
        print(f"Error creating multi-agent system: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Multi-agent collaboration example completed!")


if __name__ == "__main__":
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Error: OPENAI_API_KEY not found in environment.")
        print("Please set it in your .env file.")
        sys.exit(1)
    
    main()