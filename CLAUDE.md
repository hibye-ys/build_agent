# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multi-Model Agent Testing Framework** - a flexible agent building and testing system using LangChain and LangGraph with support for OpenAI, Anthropic, and Google models. The framework provides comprehensive testing tools, memory management, custom tools, and performance tracking capabilities with MCP (Model Context Protocol) integration.

## Essential Commands

### Development Setup
```bash
# Using uv (recommended - project uses uv.lock)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip sync

# Alternative: using pip with venv
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure environment
cp .env.example .env  # Then add your API keys
```

### Testing
```bash
pytest                           # Run all tests
pytest tests/ -v                 # Verbose test output
pytest tests/ --cov=src          # Run with coverage report
pytest tests/test_mcp_integration.py  # Run specific test file
```

### Code Quality
```bash
black .                          # Format all Python files (required before commits)
black --check .                  # Check formatting without changes
```

### Running Applications
```bash
python main.py                               # Main interactive demo with menu
python examples/basic_agent.py              # Basic agent examples
python examples/model_comparison.py         # Compare model performance
python examples/multi_agent_collaboration.py # Multi-agent workflows
python examples/mcp_basic_example.py        # MCP integration demo
python examples/mcp_advanced_example.py     # Advanced MCP features
python examples/custom_workflow.py          # Custom LangGraph workflows
python examples/prompt_management_demo.py   # Prompt versioning and A/B testing
```

## Architecture Overview

### Core Architecture
The codebase follows a modular architecture with clear separation of concerns:

- **`src/agents/`**: Different agent implementations
  - `react_agent.py`: Standard ReAct agents with tool integration
  - `enhanced_react_agent.py`: Extended ReAct with additional capabilities
  - `multi_agent.py`: Supervisor and multi-agent coordination systems
  - `memory_agent.py`: Agents with persistent conversation memory
  - `custom_agent.py`: Custom LangGraph workflow agents

- **`src/models/`**: Model provider abstractions
  - `model_factory.py`: Factory pattern for creating chat models
  - Supports OpenAI GPT-4, Anthropic Claude, Google Gemini
  - `ModelProvider` enum: `OPENAI`, `ANTHROPIC`, `GOOGLE`

- **`src/tools/`**: Custom tool system
  - `tool_registry.py`: Tool registration and management
  - `basic_tools.py`: Core tools (weather, web search, calculator, etc.)

- **`src/workflows/`**: LangGraph workflow management
  - `workflow_builder.py`: Builder pattern for complex workflows
  - `templates.py`: Pre-built workflow templates

- **`src/mcp/`**: Model Context Protocol integration
  - `manager.py`: MCP server management and coordination
  - `client.py`: MCP client implementation
  - `registry.py`: Server registration and health checking
  - `adapter.py`: LangChain tool adaptation layer
  - `config.py`: Configuration management
  - `exceptions.py`: Custom MCP exceptions
  - `resources.py`: Resource discovery and management

- **`src/prompts/`**: Advanced prompt management
  - `prompt_manager.py`: Versioning and A/B testing support
  - `validators.py`: Prompt validation and quality checks
  - `templates.py`: Reusable prompt templates
  - `loaders.py`: Prompt loading utilities

### Key Design Patterns
- **Factory Pattern**: Model creation through `ModelProvider` and `create_chat_model()`
- **Registry Pattern**: Tools and MCP servers use registration systems
- **Builder Pattern**: Workflow construction with `workflow_builder.py`
- **Configuration Objects**: Pydantic-based config classes throughout
- **Async/Await**: Consistent async patterns for I/O operations

## Development Workflow

### Task Completion Checklist
1. **Code Formatting**: Run `black .` (required before commits)
2. **Testing**: Run `pytest` (all tests must pass)
3. **Integration Testing**: Run relevant examples to verify changes
4. **Environment Variables**: Ensure `.env` has all required keys

### Code Style
- **Black formatting**: Applied to all Python code (line length 88)
- **Type hints**: Extensive use with Pydantic models
- **Async/await**: Consistent patterns for I/O operations
- **Google-style docstrings**: For classes and functions
- **Error handling**: Custom exception classes with clear messages

### Testing Approach
- **pytest** with asyncio support for async operations
- **Mocking**: Extensive use of `unittest.mock` for isolation
- **Coverage**: Maintain test coverage with `pytest --cov=src`
- **Integration tests**: Test actual model integrations when API keys available
- **Test file location**: All tests in `tests/` directory

## Key Configuration

### Environment Variables
Required API keys in `.env`:
```bash
# Required for respective model providers
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=agent-testing-framework

# Model defaults
DEFAULT_MODEL_PROVIDER=openai  # openai, anthropic, or google
DEFAULT_MODEL_NAME=gpt-4
DEFAULT_TEMPERATURE=0.7
```

### Model Usage Examples
```python
from src.models import ModelProvider, create_chat_model

# Create different model providers
openai_model = create_chat_model(ModelProvider.OPENAI, "gpt-4")
anthropic_model = create_chat_model(ModelProvider.ANTHROPIC, "claude-3-5-sonnet-20241022")
google_model = create_chat_model(ModelProvider.GOOGLE, "gemini-pro")
```

### Agent Creation Examples
```python
from src.agents import create_react_agent_wrapper, create_multi_agent_system

# Basic ReAct agent with tools
agent = create_react_agent_wrapper(
    model=openai_model,
    tools=["web_search", "calculator"],
    memory=True
)

# Multi-agent system with supervisor
multi_agent = create_multi_agent_system(
    agents={"researcher": research_config, "writer": writer_config},
    supervisor_model=anthropic_model
)
```

## Important Implementation Notes

### MCP Integration
The framework includes comprehensive MCP (Model Context Protocol) support for enhanced agent capabilities. MCP servers can be registered and managed through the `src/mcp/` module. Key components:
- Server registry for managing multiple MCP servers
- Health checking and automatic failover
- Resource discovery and caching
- LangChain tool adapter for seamless integration

### Memory Management
Agents support persistent conversation memory through LangGraph's checkpointing system. Use the `memory=True` parameter when creating agents. Memory is stored in-memory by default but can be configured for persistence.

### Workflow Customization
Complex agent behaviors can be implemented using LangGraph workflows in `src/workflows/`. The framework provides:
- Workflow builder for composing agent behaviors
- Pre-built templates for common patterns
- State management and conditional branching
- Parallel and sequential execution support

### Testing Different Models
The framework is designed for easy model comparison and testing. Use `examples/model_comparison.py` to benchmark different providers and models on various tasks. Metrics tracked include response time, token usage, and quality assessments.

### Package Management
This project uses `uv` for dependency management (as indicated by `uv.lock`). The `pyproject.toml` file defines all dependencies. If you need to add new dependencies, update `pyproject.toml` and run `uv pip sync`.