# Prompt Management System Documentation

## Overview

The Prompt Management System is a comprehensive solution for managing, versioning, and optimizing prompts in AI agent applications. It provides a centralized, reusable, and model-independent approach to prompt management.

## Features

### 🎯 Core Features

- **Template-Based Management**: Jinja2-based templating with variable substitution
- **Version Control**: Track and manage multiple versions of prompts
- **A/B Testing**: Built-in support for testing different prompt variations
- **Multi-Language Support**: Internationalization (i18n) for global applications
- **Environment-Specific Prompts**: Different prompts for dev/staging/production
- **Prompt Optimization**: Automatic token reduction while preserving meaning
- **Validation & Analysis**: Check for issues and analyze complexity
- **Metrics Tracking**: Monitor prompt performance and usage

## Installation

### Requirements

```bash
pip install jinja2 pyyaml tiktoken
```

## Architecture

```
src/prompts/
├── __init__.py              # Module exports
├── prompt_manager.py        # Core management system
├── templates.py            # Template classes
├── loaders.py             # File loaders (YAML/JSON)
└── validators.py          # Validation and optimization

prompts/templates/
├── base.yaml              # Base templates
└── i18n/                  # Language-specific templates
    ├── ko.yaml           # Korean templates
    ├── es.yaml           # Spanish templates
    └── ...
```

## Quick Start

### 1. Basic Usage

```python
from src.prompts import PromptTemplateManager

# Create manager
manager = PromptTemplateManager()

# Register a template
manager.register_template(
    id="greeting",
    name="Greeting Template",
    content="Hello {{ name }}! Welcome to {{ place }}.",
    version="1.0.0",
    variables=["name", "place"]
)

# Render the template
rendered = manager.render(
    id="greeting",
    name="Alice",
    place="AI World"
)
print(rendered)  # "Hello Alice! Welcome to AI World."
```

### 2. Using Template Classes

```python
from src.prompts import AgentPromptTemplate

# Create a ReAct agent template
template = AgentPromptTemplate.create_react_agent()

# Render with variables
prompt = template.render(
    agent_name="Assistant",
    task="Help debug Python code",
    tools=["debugger", "code_analyzer"],
    additional_context="User is a beginner"
)
```

### 3. Loading from Files

```yaml
# prompts/templates/base.yaml
prompts:
  - id: "system_default"
    name: "Default System Prompt"
    category: "system"
    template: |
      You are a helpful AI assistant.
      {% if capabilities %}
      Your capabilities include:
      {% for capability in capabilities %}
      - {{ capability }}
      {% endfor %}
      {% endif %}
```

```python
from src.prompts import PromptTemplateManager, PromptLoaderManager

manager = PromptTemplateManager()
loader = PromptLoaderManager(manager)

# Load from file
loader.load_from_file("prompts/templates/base.yaml")

# Use loaded template
prompt = manager.render(
    id="system_default",
    capabilities=["Answer questions", "Provide code"]
)
```

## Advanced Features

### Version Management

```python
# Add new version
manager.add_version(
    id="greeting",
    version="2.0.0",
    content="Greetings, {{ name }}! Welcome to {{ place }}!",
    set_current=True
)

# Rollback to previous version
manager.rollback("greeting", "1.0.0")

# Render specific version
rendered = manager.render(
    id="greeting",
    version="1.0.0",
    name="Bob",
    place="the Lab"
)
```

### A/B Testing

```python
# Create A/B test
manager.create_ab_test(
    id="assistant",
    test_name="greeting_test",
    versions=["1.0.0", "2.0.0", "3.0.0"],
    weights=[0.33, 0.33, 0.34]
)

# Renders will automatically select versions based on weights
for i in range(10):
    prompt = manager.render(id="assistant", instruction="Help user")
```

### Multi-Language Support

```python
# Register language-specific templates
manager.register_template(
    id="welcome_ko",
    name="환영 메시지",
    content="{{ name }}님, 환영합니다!",
    version="1.0.0"
)

# Render in specific language
korean_prompt = manager.render(
    id="welcome_ko",
    language="ko",
    name="사용자"
)
```

### Prompt Validation

```python
from src.prompts import PromptValidator

validator = PromptValidator(
    max_length=4000,
    max_tokens=2000,
    check_injection=True
)

result = validator.validate(prompt_text)

if result.errors:
    print("Errors:", result.errors)
if result.warnings:
    print("Warnings:", result.warnings)
if result.suggestions:
    print("Suggestions:", result.suggestions)
```

### Prompt Optimization

```python
from src.prompts import PromptOptimizer

optimizer = PromptOptimizer(target_reduction=0.3)
optimized, metrics = optimizer.optimize(
    verbose_prompt,
    preserve_meaning=True
)

print(f"Reduced by {metrics['reduction_ratio']:.1%}")
print(f"Tokens: {metrics['original_tokens']} → {metrics['optimized_tokens']}")
```

### Token Counting & Cost Estimation

```python
from src.prompts import TokenCounter

counter = TokenCounter(model="gpt-4")

# Count tokens
token_count = counter.count(prompt_text)

# Check limit
within_limit, count, limit = counter.check_limit(prompt_text)

# Estimate cost
input_cost, output_cost = counter.estimate_cost(
    prompt_text,
    expected_output_tokens=500
)
```

## Integration with Agents

### Enhanced ReAct Agent

```python
from src.agents.enhanced_react_agent import create_enhanced_react_agent

# Create agent with prompt management
agent = create_enhanced_react_agent(
    model_provider="openai",
    model_name="gpt-4",
    tool_names=["calculator", "web_search"],
    prompt_id="react_agent",  # Use template ID
    language="en",
    memory=True,
    environment="production"
)

# Agent automatically handles:
# - Prompt loading and rendering
# - Version selection
# - Validation
# - Optimization
# - Metrics tracking

response = agent.invoke({"messages": [...]})
```

### Switching Languages Dynamically

```python
# Start with English
agent = create_enhanced_react_agent(language="en")

# Switch to Korean
agent.switch_language("ko")

# Switch to Spanish
agent.switch_language("es")
```

### Updating Prompt Versions

```python
# Update to specific version
agent.update_prompt_version("2.0.0")

# Get current metrics
metrics = agent.get_prompt_metrics()
print(f"Success rate: {metrics['success_rate']}")
print(f"Avg tokens: {metrics['avg_tokens_used']}")
```

## Template Structure

### YAML Format

```yaml
version: "1.0"
created_at: "2024-01-01"
description: "Template collection description"

prompts:
  - id: "unique_id"
    name: "Human Readable Name"
    category: "system|agent|user|specialist"
    description: "What this prompt does"
    tags: ["tag1", "tag2"]
    variables: ["var1", "var2"]
    template: |
      Your Jinja2 template here.
      Variable: {{ var1 }}
      {% if var2 %}
      Optional: {{ var2 }}
      {% endif %}
    versions:
      "1.0.0":
        content: "..."
        current: true
      "2.0.0":
        content: "..."
        current: false
```

### Composite Templates

```python
from src.prompts import CompositePromptTemplate

# Combine multiple templates
composite = CompositePromptTemplate("full_agent")

# Add components
composite.add_component(system_template)
composite.add_component(agent_template, separator="\n---\n")
composite.add_component(task_template)

# Render all at once
full_prompt = composite.render(**all_variables)
```

## Environment Configuration

### Environment-Specific Loading

```python
from src.prompts import EnvironmentPromptLoader, PromptEnvironment

loader = EnvironmentPromptLoader(
    base_dir="prompts/templates",
    environment=PromptEnvironment.PRODUCTION
)

# Automatically loads:
# 1. base.yaml (shared templates)
# 2. production.yaml (environment-specific)
# 3. production/*.yaml (environment directory)

count = loader.load_for_environment(manager)
```

### File Structure for Environments

```
prompts/templates/
├── base.yaml              # Shared across all environments
├── development.yaml       # Development-specific
├── staging.yaml          # Staging-specific
├── production.yaml       # Production-specific
├── development/
│   └── debug.yaml       # Additional dev templates
├── staging/
│   └── test.yaml        # Additional staging templates
└── production/
    └── optimized.yaml   # Additional prod templates
```

## Best Practices

### 1. Template Organization

- Use meaningful IDs and names
- Categorize templates properly
- Tag templates for easy filtering
- Document all variables

### 2. Version Management

- Use semantic versioning (major.minor.patch)
- Document changes in version metadata
- Test new versions before setting as current
- Keep rollback versions available

### 3. Performance Optimization

- Cache frequently used templates
- Optimize verbose prompts for production
- Monitor token usage and costs
- Use appropriate model limits

### 4. Security

- Validate all user inputs
- Check for injection attempts
- Sanitize template variables
- Use environment variables for sensitive data

### 5. Testing

- Unit test prompt templates
- A/B test variations
- Monitor success rates
- Collect user feedback

## Metrics and Monitoring

```python
# Export metrics
metrics = manager.export_metrics(output_path="metrics.json")

# Analyze specific template
template = manager.get_template("react_agent")
metrics = template.get_metrics("1.0.0")

print(f"Usage count: {metrics.usage_count}")
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Avg response time: {metrics.avg_response_time:.2f}s")
print(f"Avg tokens: {metrics.avg_tokens_used}")
print(f"Avg feedback: {metrics.avg_feedback or 'N/A'}")
```

## Migration Guide

### From Hardcoded Prompts

Before:
```python
agent = create_react_agent(
    system_prompt="You are a helpful assistant..."
)
```

After:
```python
agent = create_enhanced_react_agent(
    prompt_id="react_agent",
    prompt_version="2.0.0",
    language="en"
)
```

### Benefits of Migration

1. **Centralized Management**: All prompts in one place
2. **Version Control**: Track changes over time
3. **Reusability**: Share prompts across agents
4. **Optimization**: Automatic token reduction
5. **Testing**: Built-in A/B testing
6. **Internationalization**: Multi-language support
7. **Metrics**: Performance tracking

## Troubleshooting

### Common Issues

1. **Template Not Found**
   - Check template ID spelling
   - Verify file was loaded
   - Check environment settings

2. **Missing Variables**
   - Review template variables list
   - Provide all required variables
   - Check for typos in variable names

3. **Token Limit Exceeded**
   - Use prompt optimization
   - Reduce template verbosity
   - Consider splitting into smaller prompts

4. **Language Not Available**
   - Verify language template exists
   - Check file naming convention
   - Fallback to default language

## Examples

See `examples/prompt_management_demo.py` for comprehensive examples including:

- Basic template management
- Different template types
- Composite prompts
- Validation and optimization
- Loading from files
- A/B testing
- Multi-language support
- Enhanced agent integration

## Contributing

When adding new features:

1. Follow existing code structure
2. Add unit tests
3. Update documentation
4. Provide examples
5. Test with multiple models

## License

This prompt management system is part of the LangChain/LangGraph agent framework and follows the same licensing terms.