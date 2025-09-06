"""Base template classes for different types of prompts."""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from jinja2 import Template, Environment, meta
import re


@dataclass
class BasePromptTemplate(ABC):
    """Base class for all prompt templates."""
    
    name: str
    template: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract variables from template if not provided."""
        if not self.variables:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract variable names from Jinja2 template."""
        env = Environment()
        ast = env.parse(self.template)
        return list(meta.find_undeclared_variables(ast))
    
    @abstractmethod
    def validate_variables(self, **kwargs) -> bool:
        """Validate that all required variables are provided."""
        pass
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        if not self.validate_variables(**kwargs):
            missing = set(self.variables) - set(kwargs.keys())
            raise ValueError(f"Missing required variables: {missing}")
        
        template = Template(self.template)
        return template.render(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "variables": self.variables,
            "metadata": self.metadata
        }


@dataclass
class SystemPromptTemplate(BasePromptTemplate):
    """Template for system prompts."""
    
    role: str = "system"
    capabilities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def validate_variables(self, **kwargs) -> bool:
        """Validate system prompt variables."""
        # System prompts often don't require variables
        required = [v for v in self.variables if v not in self.metadata.get("optional", [])]
        return all(v in kwargs for v in required)
    
    @classmethod
    def create_default(cls) -> "SystemPromptTemplate":
        """Create a default system prompt template."""
        template = """You are a helpful AI assistant.
{% if capabilities %}
Your capabilities include:
{% for capability in capabilities %}
- {{ capability }}
{% endfor %}
{% endif %}

{% if constraints %}
Please adhere to these constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

{% if custom_instructions %}
{{ custom_instructions }}
{% endif %}"""
        
        return cls(
            name="default_system",
            template=template,
            description="Default system prompt for general assistance",
            capabilities=["Answering questions", "Providing explanations", "Following instructions"],
            constraints=["Be helpful and harmless", "Provide accurate information"]
        )


@dataclass
class AgentPromptTemplate(BasePromptTemplate):
    """Template for agent-specific prompts."""
    
    agent_type: str = ""
    tools: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    
    def validate_variables(self, **kwargs) -> bool:
        """Validate agent prompt variables."""
        required_vars = ["agent_name", "task"]
        return all(v in kwargs or v in self.variables for v in required_vars)
    
    @classmethod
    def create_react_agent(cls) -> "AgentPromptTemplate":
        """Create a ReAct agent prompt template."""
        template = """You are {{ agent_name }}, an AI agent with access to various tools.

Your task: {{ task }}

{% if tools %}
Available tools:
{% for tool in tools %}
- {{ tool }}
{% endfor %}
{% endif %}

Think step by step:
1. Understand what information or action is needed
2. Decide which tool(s) to use
3. Use the tools to gather information or perform actions
4. Analyze the results
5. Provide a comprehensive response

{% if additional_context %}
Additional context:
{{ additional_context }}
{% endif %}

Remember to:
- Use tools when they can provide accurate information
- Think critically about the results
- Provide clear and helpful responses"""
        
        return cls(
            name="react_agent",
            template=template,
            description="Template for ReAct pattern agents",
            agent_type="react"
        )
    
    @classmethod
    def create_specialist_agent(cls, specialty: str) -> "AgentPromptTemplate":
        """Create a specialist agent template."""
        templates = {
            "research": """You are a research specialist focused on {{ task }}.

Your approach:
1. Identify key information needs
2. Search for relevant data using available tools
3. Analyze and synthesize findings
4. Present conclusions with evidence

{% if sources %}
Priority sources:
{% for source in sources %}
- {{ source }}
{% endfor %}
{% endif %}

Maintain objectivity and cite sources when possible.""",
            
            "writer": """You are a professional writer working on {{ task }}.

Writing guidelines:
- Style: {{ style | default('clear and engaging') }}
- Tone: {{ tone | default('professional') }}
- Length: {{ length | default('appropriate to task') }}

{% if audience %}
Target audience: {{ audience }}
{% endif %}

Focus on clarity, coherence, and impact.""",
            
            "analyst": """You are a data analyst examining {{ task }}.

Analysis framework:
1. Data collection and validation
2. Statistical analysis and pattern recognition
3. Insight generation
4. Actionable recommendations

{% if metrics %}
Key metrics to consider:
{% for metric in metrics %}
- {{ metric }}
{% endfor %}
{% endif %}

Provide data-driven insights with clear visualizations when possible."""
        }
        
        template = templates.get(specialty, templates["research"])
        
        return cls(
            name=f"{specialty}_agent",
            template=template,
            description=f"Template for {specialty} specialist agents",
            agent_type=specialty
        )


@dataclass
class UserPromptTemplate(BasePromptTemplate):
    """Template for user prompts."""
    
    input_format: str = "text"
    output_format: str = "text"
    examples: List[Dict[str, str]] = field(default_factory=list)
    
    def validate_variables(self, **kwargs) -> bool:
        """Validate user prompt variables."""
        return "user_input" in kwargs or "query" in kwargs
    
    @classmethod
    def create_qa_template(cls) -> "UserPromptTemplate":
        """Create a question-answering template."""
        template = """Question: {{ query }}

{% if context %}
Context:
{{ context }}
{% endif %}

{% if examples %}
Examples:
{% for example in examples %}
Q: {{ example.question }}
A: {{ example.answer }}
{% endfor %}
{% endif %}

Please provide a {{ response_type | default('detailed') }} answer."""
        
        return cls(
            name="qa_template",
            template=template,
            description="Template for question-answering interactions",
            input_format="question",
            output_format="answer"
        )
    
    @classmethod
    def create_task_template(cls) -> "UserPromptTemplate":
        """Create a task execution template."""
        template = """Task: {{ task_description }}

{% if requirements %}
Requirements:
{% for req in requirements %}
- {{ req }}
{% endfor %}
{% endif %}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

{% if expected_output %}
Expected output format:
{{ expected_output }}
{% endif %}

Please complete this task following all specified requirements."""
        
        return cls(
            name="task_template",
            template=template,
            description="Template for task execution requests",
            input_format="task",
            output_format="result"
        )


class PromptTemplateFactory:
    """Factory for creating prompt templates."""
    
    @staticmethod
    def create_template(
        template_type: str,
        name: str,
        **kwargs
    ) -> BasePromptTemplate:
        """Create a prompt template of the specified type.
        
        Args:
            template_type: Type of template (system, agent, user)
            name: Template name
            **kwargs: Additional parameters for template
            
        Returns:
            Created template instance
        """
        template_classes = {
            "system": SystemPromptTemplate,
            "agent": AgentPromptTemplate,
            "user": UserPromptTemplate
        }
        
        template_class = template_classes.get(template_type)
        if not template_class:
            raise ValueError(f"Unknown template type: {template_type}")
        
        # Handle special creation methods
        if template_type == "system" and name == "default":
            return SystemPromptTemplate.create_default()
        elif template_type == "agent":
            if name == "react":
                return AgentPromptTemplate.create_react_agent()
            elif name in ["research", "writer", "analyst"]:
                return AgentPromptTemplate.create_specialist_agent(name)
        elif template_type == "user":
            if name == "qa":
                return UserPromptTemplate.create_qa_template()
            elif name == "task":
                return UserPromptTemplate.create_task_template()
        
        # Default creation
        return template_class(name=name, **kwargs)


class CompositePromptTemplate:
    """Combines multiple templates into a complex prompt."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize composite template.
        
        Args:
            name: Template name
            description: Template description
        """
        self.name = name
        self.description = description
        self.components: List[BasePromptTemplate] = []
        self.separators: Dict[int, str] = {}
    
    def add_component(
        self,
        template: BasePromptTemplate,
        separator: str = "\n\n"
    ) -> "CompositePromptTemplate":
        """Add a component template.
        
        Args:
            template: Template to add
            separator: Separator before this component
            
        Returns:
            Self for chaining
        """
        if self.components:
            self.separators[len(self.components)] = separator
        self.components.append(template)
        return self
    
    def render(self, **kwargs) -> str:
        """Render all component templates.
        
        Args:
            **kwargs: Variables for all templates
            
        Returns:
            Combined rendered prompt
        """
        rendered_parts = []
        
        for i, component in enumerate(self.components):
            if i > 0 and i in self.separators:
                rendered_parts.append(self.separators[i])
            
            # Filter kwargs for this component
            component_kwargs = {
                k: v for k, v in kwargs.items()
                if k in component.variables or k in ["capabilities", "constraints", "tools"]
            }
            
            rendered_parts.append(component.render(**component_kwargs))
        
        return "".join(rendered_parts)
    
    def get_all_variables(self) -> List[str]:
        """Get all variables from all components.
        
        Returns:
            List of unique variable names
        """
        all_vars = set()
        for component in self.components:
            all_vars.update(component.variables)
        return list(all_vars)