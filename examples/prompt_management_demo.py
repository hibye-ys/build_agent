"""Demonstration of the prompt management system."""

import os
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.prompts import (
    PromptTemplateManager,
    PromptTemplate,
    PromptLoaderManager,
    PromptValidator,
    TokenCounter,
    PromptOptimizer,
    PromptEnvironment,
    AgentPromptTemplate,
    SystemPromptTemplate,
    CompositePromptTemplate
)
from src.agents.enhanced_react_agent import create_enhanced_react_agent


def demo_basic_prompt_management():
    """Demonstrate basic prompt template management."""
    print("\n" + "="*60)
    print("DEMO: Basic Prompt Management")
    print("="*60)
    
    # Create prompt manager
    manager = PromptTemplateManager(
        environment=PromptEnvironment.DEVELOPMENT,
        default_language="en"
    )
    
    # Register a simple template
    template = manager.register_template(
        id="greeting",
        name="Greeting Template",
        content="Hello {{ name }}! Welcome to {{ place }}.",
        description="A simple greeting template",
        category="general",
        version="1.0.0",
        variables=["name", "place"]
    )
    
    # Render the template
    rendered = manager.render(
        id="greeting",
        name="Alice",
        place="AI World"
    )
    print(f"\nRendered template: {rendered}")
    
    # Add a new version
    manager.add_version(
        id="greeting",
        version="2.0.0",
        content="Greetings, {{ name }}! You've arrived at {{ place }}. Enjoy your stay!",
        set_current=True
    )
    
    # Render with new version
    rendered_v2 = manager.render(
        id="greeting",
        name="Bob",
        place="the Future"
    )
    print(f"Version 2.0.0: {rendered_v2}")
    
    # Rollback to previous version
    manager.rollback("greeting", "1.0.0")
    rendered_v1 = manager.render(
        id="greeting",
        name="Charlie",
        place="the Lab"
    )
    print(f"After rollback: {rendered_v1}")


def demo_prompt_templates():
    """Demonstrate different types of prompt templates."""
    print("\n" + "="*60)
    print("DEMO: Prompt Template Types")
    print("="*60)
    
    # System prompt template
    system_prompt = SystemPromptTemplate.create_default()
    rendered_system = system_prompt.render(
        capabilities=["Answer questions", "Provide code examples", "Explain concepts"],
        constraints=["Be helpful", "Be accurate", "Be concise"],
        custom_instructions="Focus on Python programming."
    )
    print("\nSystem Prompt:")
    print(rendered_system)
    
    # Agent prompt template
    react_prompt = AgentPromptTemplate.create_react_agent()
    rendered_agent = react_prompt.render(
        agent_name="CodeBot",
        task="Help debug a Python script",
        tools=["debugger", "code_analyzer", "documentation_search"],
        additional_context="The user is a beginner programmer."
    )
    print("\nReAct Agent Prompt:")
    print(rendered_agent[:300] + "...")  # Show first 300 chars
    
    # Specialist agent prompt
    research_prompt = AgentPromptTemplate.create_specialist_agent("research")
    rendered_research = research_prompt.render(
        task="Research best practices for API design",
        sources=["Official documentation", "Industry standards", "Academic papers"]
    )
    print("\nResearch Specialist Prompt:")
    print(rendered_research[:300] + "...")


def demo_composite_prompts():
    """Demonstrate composite prompt templates."""
    print("\n" + "="*60)
    print("DEMO: Composite Prompts")
    print("="*60)
    
    # Create a composite prompt
    composite = CompositePromptTemplate(
        name="full_agent",
        description="Complete agent prompt with system and task components"
    )
    
    # Add system component
    system = SystemPromptTemplate.create_default()
    composite.add_component(system, separator="\n\n---\n\n")
    
    # Add agent component
    agent = AgentPromptTemplate.create_react_agent()
    composite.add_component(agent)
    
    # Render the composite
    rendered = composite.render(
        # System variables
        capabilities=["Use tools", "Reason step by step"],
        constraints=["Be helpful", "Be accurate"],
        # Agent variables
        agent_name="Assistant",
        task="Solve complex problems",
        tools=["calculator", "web_search"]
    )
    
    print("\nComposite Prompt:")
    print(rendered[:500] + "...")
    print(f"\nTotal variables: {composite.get_all_variables()}")


def demo_prompt_validation():
    """Demonstrate prompt validation and optimization."""
    print("\n" + "="*60)
    print("DEMO: Prompt Validation & Optimization")
    print("="*60)
    
    # Create a verbose prompt
    verbose_prompt = """
    You are a very very helpful AI assistant. Please please help the user.
    
    In order to assist you better, I need to understand your request.
    At this point in time, I can help with various tasks.
    Due to the fact that I have access to tools, I can provide accurate information.
    
    
    
    Please provide your request and I will really really try to help you.
    """
    
    # Validate the prompt
    validator = PromptValidator(max_length=4000, max_tokens=2000)
    result = validator.validate(verbose_prompt)
    
    print("\nValidation Results:")
    print(f"Valid: {result.is_valid}")
    print(f"Warnings: {result.warnings}")
    print(f"Suggestions: {result.suggestions}")
    print(f"Token count: {result.metrics.get('token_count', 'N/A')}")
    
    # Optimize the prompt
    optimizer = PromptOptimizer(target_reduction=0.3)
    optimized, metrics = optimizer.optimize(verbose_prompt)
    
    print("\nOptimization Results:")
    print(f"Original tokens: {metrics['original_tokens']}")
    print(f"Optimized tokens: {metrics['optimized_tokens']}")
    print(f"Reduction: {metrics['reduction_ratio']:.1%}")
    print(f"\nOptimized prompt:")
    print(optimized)


def demo_prompt_loading():
    """Demonstrate loading prompts from files."""
    print("\n" + "="*60)
    print("DEMO: Loading Prompts from Files")
    print("="*60)
    
    # Create manager
    manager = PromptTemplateManager(
        templates_dir=Path("prompts/templates"),
        environment=PromptEnvironment.DEVELOPMENT
    )
    
    # Create loader
    loader = PromptLoaderManager(manager)
    
    # Try to load from base file
    base_file = Path("prompts/templates/base.yaml")
    if base_file.exists():
        count = loader.load_from_file(base_file)
        print(f"\nLoaded {count} prompts from base.yaml")
        
        # List loaded templates
        templates = manager.list_templates()
        print(f"\nAvailable templates:")
        for template in templates[:5]:  # Show first 5
            print(f"  - {template.id}: {template.name} ({template.category})")
        
        # Render a loaded template
        if templates:
            first_template = templates[0]
            try:
                # Prepare variables based on template requirements
                variables = {}
                if "agent_name" in first_template.variables:
                    variables["agent_name"] = "Assistant"
                if "task" in first_template.variables:
                    variables["task"] = "Help the user"
                if "tools" in first_template.variables:
                    variables["tools"] = ["calculator", "web_search"]
                
                rendered = manager.render(
                    id=first_template.id,
                    **variables
                )
                print(f"\nRendered '{first_template.id}':")
                print(rendered[:300] + "...")
            except Exception as e:
                print(f"Could not render template: {e}")
    else:
        print("No template files found. Creating example templates...")
        
        # Create some example templates
        manager.register_template(
            id="example",
            name="Example Template",
            content="This is an example: {{ example_var }}",
            description="Just an example",
            version="1.0.0",
            variables=["example_var"]
        )
        print("Created example template")


def demo_ab_testing():
    """Demonstrate A/B testing with prompts."""
    print("\n" + "="*60)
    print("DEMO: A/B Testing")
    print("="*60)
    
    # Create manager
    manager = PromptTemplateManager()
    
    # Register a template with multiple versions
    template = manager.register_template(
        id="assistant",
        name="Assistant Prompt",
        content="You are a helpful assistant. {{ instruction }}",
        version="1.0.0",
        variables=["instruction"]
    )
    
    manager.add_version(
        id="assistant",
        version="2.0.0",
        content="You are an AI assistant ready to help. {{ instruction }}",
        set_current=False
    )
    
    manager.add_version(
        id="assistant",
        version="3.0.0",
        content="I'm your AI assistant. {{ instruction }}",
        set_current=False
    )
    
    # Create A/B test
    test_config = manager.create_ab_test(
        id="assistant",
        test_name="assistant_greeting_test",
        versions=["1.0.0", "2.0.0", "3.0.0"],
        weights=[0.33, 0.33, 0.34]
    )
    
    print(f"\nCreated A/B test: {test_config['test_name']}")
    print(f"Testing versions: {test_config['versions']}")
    
    # Simulate multiple renders to see distribution
    print("\nSimulating 10 renders:")
    for i in range(10):
        rendered = manager.render(
            id="assistant",
            instruction="Please be concise.",
            record_metrics=False
        )
        print(f"  Render {i+1}: {rendered[:30]}...")
    
    print(f"\nSelection counts: {test_config['selection_count']}")


def demo_multilanguage_support():
    """Demonstrate multi-language prompt support."""
    print("\n" + "="*60)
    print("DEMO: Multi-Language Support")
    print("="*60)
    
    # Create manager
    manager = PromptTemplateManager(default_language="en")
    
    # Register English template
    manager.register_template(
        id="welcome",
        name="Welcome Message",
        content="Welcome {{ name }}! How can I help you today?",
        version="1.0.0",
        variables=["name"]
    )
    
    # Register Korean version
    manager.register_template(
        id="welcome_ko",
        name="환영 메시지",
        content="{{ name }}님, 환영합니다! 오늘 무엇을 도와드릴까요?",
        version="1.0.0",
        variables=["name"]
    )
    
    # Register Spanish version
    manager.register_template(
        id="welcome_es",
        name="Mensaje de Bienvenida",
        content="¡Bienvenido {{ name }}! ¿Cómo puedo ayudarte hoy?",
        version="1.0.0",
        variables=["name"]
    )
    
    # Render in different languages
    languages = [
        ("en", "welcome"),
        ("ko", "welcome_ko"),
        ("es", "welcome_es")
    ]
    
    print("\nWelcome messages in different languages:")
    for lang, template_id in languages:
        try:
            rendered = manager.render(
                id=template_id,
                language=lang,
                name="User"
            )
            print(f"  {lang}: {rendered}")
        except Exception as e:
            print(f"  {lang}: Error - {e}")


def demo_enhanced_agent():
    """Demonstrate the enhanced ReAct agent with prompt management."""
    print("\n" + "="*60)
    print("DEMO: Enhanced ReAct Agent")
    print("="*60)
    
    try:
        # Create an enhanced agent
        agent = create_enhanced_react_agent(
            model_provider="openai",
            model_name="gpt-4",
            tool_names=["calculator", "datetime"],
            prompt_id="react_agent",
            language="en",
            memory=True,
            environment="development"
        )
        
        print("\n✅ Enhanced agent created successfully!")
        
        # Get prompt metrics
        metrics = agent.get_prompt_metrics()
        if metrics:
            print(f"\nPrompt metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        # Switch language (example)
        print("\nSwitching to Korean prompts...")
        # agent.switch_language("ko")
        print("(Language switch simulated - would require Korean templates)")
        
        # Update version (example)
        print("\nUpdating to version 2.0.0...")
        # agent.update_prompt_version("2.0.0")
        print("(Version update simulated - would require version 2.0.0)")
        
    except Exception as e:
        print(f"\n⚠️ Could not create enhanced agent: {e}")
        print("This is expected if API keys are not configured.")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print(" PROMPT MANAGEMENT SYSTEM DEMONSTRATION")
    print("="*80)
    
    demos = [
        ("Basic Prompt Management", demo_basic_prompt_management),
        ("Prompt Template Types", demo_prompt_templates),
        ("Composite Prompts", demo_composite_prompts),
        ("Validation & Optimization", demo_prompt_validation),
        ("Loading from Files", demo_prompt_loading),
        ("A/B Testing", demo_ab_testing),
        ("Multi-Language Support", demo_multilanguage_support),
        ("Enhanced Agent", demo_enhanced_agent)
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
    
    print("\n" + "="*80)
    print(" DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()