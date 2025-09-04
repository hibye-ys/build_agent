"""Memory-enabled agent implementation."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledStateGraph

from ..models import ModelConfig, create_chat_model
from ..tools import get_tools_by_names


class PersistentMemorySaver(SqliteSaver):
    """Persistent memory saver using SQLite."""
    
    def __init__(self, db_path: str = "checkpoints.db"):
        """Initialize persistent memory saver.
        
        Args:
            db_path: Path to SQLite database file
        """
        # Ensure parent directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize with connection string
        super().__init__(conn_string=str(db_file))


def create_memory_agent(
    model_config: ModelConfig,
    tools: List[str],
    system_prompt: Optional[str] = None,
    memory_type: str = "in_memory",
    memory_config: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None
) -> CompiledStateGraph:
    """Create an agent with memory capabilities.
    
    Args:
        model_config: Model configuration
        tools: List of tool names
        system_prompt: Optional system prompt
        memory_type: Type of memory ("in_memory" or "persistent")
        memory_config: Additional memory configuration
        conversation_id: Optional conversation ID for persistence
        
    Returns:
        Compiled agent graph with memory
        
    Example:
        >>> agent = create_memory_agent(
        ...     model_config=ModelConfig(
        ...         provider=ModelProvider.OPENAI,
        ...         model_name="gpt-4"
        ...     ),
        ...     tools=["calculator", "web_search"],
        ...     memory_type="persistent",
        ...     conversation_id="user_123"
        ... )
    """
    # Create model
    model = create_chat_model(model_config)
    
    # Get tools
    tool_instances = get_tools_by_names(tools)
    
    # Create memory saver
    if memory_type == "in_memory":
        checkpointer = MemorySaver()
    elif memory_type == "persistent":
        db_path = "checkpoints.db"
        if memory_config and "db_path" in memory_config:
            db_path = memory_config["db_path"]
        checkpointer = PersistentMemorySaver(db_path=db_path)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
    
    # Create agent with memory
    agent = create_react_agent(
        model=model,
        tools=tool_instances,
        prompt=system_prompt,
        checkpointer=checkpointer
    )
    
    return agent


class ConversationManager:
    """Manage conversations with persistent memory."""
    
    def __init__(self, db_path: str = "conversations.db"):
        """Initialize conversation manager.
        
        Args:
            db_path: Path to conversation database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpointer = PersistentMemorySaver(db_path=str(self.db_path))
    
    def create_agent(
        self,
        conversation_id: str,
        model_config: ModelConfig,
        tools: List[str],
        system_prompt: Optional[str] = None
    ) -> CompiledStateGraph:
        """Create an agent for a specific conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            model_config: Model configuration
            tools: List of tool names
            system_prompt: Optional system prompt
            
        Returns:
            Agent with conversation-specific memory
        """
        model = create_chat_model(model_config)
        tool_instances = get_tools_by_names(tools)
        
        agent = create_react_agent(
            model=model,
            tools=tool_instances,
            prompt=system_prompt,
            checkpointer=self.checkpointer
        )
        
        return agent
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        # This would need to be implemented based on the checkpointer's
        # actual storage mechanism
        return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation's history.
        
        Args:
            conversation_id: Conversation to delete
            
        Returns:
            True if deleted, False otherwise
        """
        # This would need to be implemented based on the checkpointer's
        # actual storage mechanism
        return False
    
    def export_conversation(
        self,
        conversation_id: str,
        output_path: str
    ) -> bool:
        """Export conversation history to file.
        
        Args:
            conversation_id: Conversation to export
            output_path: Path to output file
            
        Returns:
            True if exported, False otherwise
        """
        # This would need to be implemented based on the checkpointer's
        # actual storage mechanism
        return False