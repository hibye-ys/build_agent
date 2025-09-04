# Multi-Model Agent Testing Framework

A flexible agent building and testing framework using LangChain and LangGraph with support for OpenAI, Anthropic, and Google models.

## Features

- 🤖 **Multi-Model Support**: Seamlessly switch between OpenAI, Anthropic, and Google models
- 🔄 **Flexible Agent Flows**: Build custom workflows using LangGraph
- 🧪 **Testing Environment**: Comprehensive testing tools for agent evaluation
- 💾 **Memory Management**: Persistent conversation history and state management
- 🛠️ **Custom Tools**: Extensible tool system for agent capabilities
- 📊 **Performance Tracking**: Monitor and compare model performance

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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
from src.agents import create_agent
from src.models import ModelProvider

# Create an agent with OpenAI
agent = create_agent(
    provider=ModelProvider.OPENAI,
    model="gpt-4",
    tools=["web_search", "calculator"],
    memory=True
)

# Run the agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Seoul?"}]
})
```

## Project Structure

```
├── src/
│   ├── agents/       # Agent implementations
│   ├── models/       # Model configurations
│   ├── tools/        # Custom tools
│   ├── workflows/    # LangGraph workflows
│   └── utils/        # Helper functions
├── tests/            # Test scenarios
├── examples/         # Example usage
├── config/           # Configuration files
└── docs/             # Documentation
```

## Examples

See the `examples/` directory for detailed usage examples:
- Basic agent usage
- Multi-agent collaboration
- Custom workflow creation
- Model comparison testing

## Development

```bash
# Run tests
pytest

# Format code
black .

# Run examples
python examples/basic_agent.py
```

## License

MIT