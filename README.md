# Multi-Model Agent Testing Framework

A flexible agent building and testing framework using LangChain and LangGraph with support for OpenAI, Anthropic, and Google models. The framework provides comprehensive testing tools, memory management, custom tools, and performance tracking capabilities with MCP (Model Context Protocol) integration.

## Features

- 🤖 **Multi-Model Support**: Seamlessly switch between OpenAI, Anthropic, and Google models
- 🔄 **Flexible Agent Flows**: Build custom workflows using LangGraph
- 🧪 **Testing Environment**: Comprehensive testing tools for agent evaluation
- 💾 **Memory Management**: Persistent conversation history and state management
- 🛠️ **Custom Tools**: Extensible tool system for agent capabilities
- 📊 **Performance Tracking**: Monitor and compare model performance
- 🔌 **MCP Integration**: Model Context Protocol support for enhanced capabilities
- 📝 **Prompt Management**: Advanced prompt versioning and A/B testing

## Quick Start

### Installation

**Using uv (recommended):**
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip sync
```

**Alternative using pip:**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install as editable package
pip install -e .
```

### Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### Basic Usage

```python
from src.agents import create_react_agent_wrapper
from src.models import ModelProvider, create_chat_model

# Create a chat model
model = create_chat_model(ModelProvider.OPENAI, "gpt-4")

# Create an agent with tools and memory
agent = create_react_agent_wrapper(
    model=model,
    tools=["web_search", "calculator"],
    memory=True
)

# Run the agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Seoul?"}]
})
```

### Interactive Demo

Run the main interactive demo:
```bash
python main.py
```

## Project Structure

```
├── src/
│   ├── agents/       # Agent implementations (ReAct, Enhanced ReAct, Multi-agent, Memory, Custom)
│   ├── models/       # Model provider abstractions and factory pattern
│   ├── tools/        # Custom tool system with registry
│   ├── workflows/    # LangGraph workflow management and templates
│   ├── mcp/          # Model Context Protocol integration
│   │   ├── manager.py    # MCP server management and coordination
│   │   ├── client.py     # MCP client implementation
│   │   ├── registry.py   # Server registration and health checking
│   │   └── adapter.py    # LangChain tool adaptation layer
│   ├── prompts/      # Advanced prompt management
│   │   ├── prompt_manager.py  # Versioning and A/B testing
│   │   ├── validators.py      # Prompt validation
│   │   └── templates.py       # Reusable templates
│   └── utils/        # Helper functions
├── tests/            # Comprehensive test scenarios
├── examples/         # Usage examples and demos
├── main.py          # Interactive demo application
├── pyproject.toml   # Project dependencies and configuration
└── uv.lock          # Dependency lock file (uv package manager)
```

## Examples

See the `examples/` directory for detailed usage examples:

- **`basic_agent.py`** - Basic agent examples with different models
- **`multi_agent_collaboration.py`** - Multi-agent workflows and supervision
- **`model_comparison.py`** - Compare performance across different models
- **`mcp_basic_example.py`** - MCP integration demonstration
- **`mcp_advanced_example.py`** - Advanced MCP features and capabilities
- **`custom_workflow.py`** - Custom LangGraph workflows
- **`prompt_management_demo.py`** - Prompt versioning and A/B testing

## Development

### Testing and Code Quality
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src

# Format all Python files (required before commits)
black .

# Check formatting without changes
black --check .
```

### Running Examples
```bash
python main.py                               # Interactive demo with menu
python examples/basic_agent.py              # Basic agent examples
python examples/model_comparison.py         # Model performance comparison
python examples/multi_agent_collaboration.py # Multi-agent workflows
python examples/mcp_basic_example.py        # MCP integration demo
```

### Environment Configuration

Optional environment variables for enhanced functionality:
```bash
# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=agent-testing-framework

# Model defaults
DEFAULT_MODEL_PROVIDER=openai  # openai, anthropic, or google
DEFAULT_MODEL_NAME=gpt-4
DEFAULT_TEMPERATURE=0.7
```

## License

MIT