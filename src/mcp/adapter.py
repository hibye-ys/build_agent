"""
LangChain adapter for MCP tools integration.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Type, Union, Callable
from abc import ABC, abstractmethod
from functools import wraps
from pydantic import BaseModel, Field, create_model

from langchain.tools import BaseTool, StructuredTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun
)
from langchain.pydantic_v1 import BaseModel as LangChainBaseModel, Field as LangChainField

from .client import MCPClient, MCPTool
from .exceptions import MCPToolConversionError


logger = logging.getLogger(__name__)


def create_pydantic_model_from_schema(
    name: str,
    schema: Dict[str, Any],
    description: Optional[str] = None
) -> Type[BaseModel]:
    """Create a Pydantic model from JSON schema.
    
    Args:
        name: Model name
        schema: JSON schema
        description: Model description
        
    Returns:
        Pydantic model class
    """
    if not schema or schema.get('type') != 'object':
        # Return empty model for tools with no parameters
        return create_model(f"{name}Input", __base__=LangChainBaseModel)
    
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    fields = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get('type', 'string')
        prop_description = prop_schema.get('description', '')
        
        # Map JSON schema types to Python types
        type_mapping = {
            'string': str,
            'number': float,
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        python_type = type_mapping.get(prop_type, Any)
        
        # Handle arrays with items
        if prop_type == 'array' and 'items' in prop_schema:
            item_type = prop_schema['items'].get('type', 'string')
            if item_type in type_mapping:
                python_type = List[type_mapping[item_type]]
        
        # Create field
        if prop_name in required:
            fields[prop_name] = (python_type, LangChainField(description=prop_description))
        else:
            default_value = prop_schema.get('default', None)
            fields[prop_name] = (Optional[python_type], LangChainField(default=default_value, description=prop_description))
    
    # Create and return the model
    return create_model(
        f"{name}Input",
        __base__=LangChainBaseModel,
        __doc__=description,
        **fields
    )


class MCPBaseTool(BaseTool):
    """Base class for MCP tools in LangChain."""
    
    mcp_client: MCPClient = Field(exclude=True)
    mcp_tool_name: str = Field(exclude=True)
    handle_tool_error: bool = True
    
    def _run(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """Run tool synchronously (not supported for MCP tools)."""
        raise NotImplementedError("MCP tools only support async operations. Use arun instead.")
    
    async def _arun(
        self,
        *args,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """Run tool asynchronously."""
        try:
            # Call MCP tool
            result = await self.mcp_client.call_tool(self.mcp_tool_name, kwargs)
            
            # Process result
            if isinstance(result, list):
                # Handle multiple content items
                processed_results = []
                for item in result:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            processed_results.append(item.get('text', ''))
                        elif item.get('type') == 'image':
                            # Handle image data
                            processed_results.append(f"[Image: {item.get('data', '')[:50]}...]")
                        else:
                            processed_results.append(json.dumps(item))
                    else:
                        processed_results.append(str(item))
                
                return '\n'.join(processed_results)
            else:
                return str(result)
                
        except Exception as e:
            if self.handle_tool_error:
                return f"Tool execution failed: {str(e)}"
            raise


class MCPToolConverter:
    """Converts MCP tools to LangChain tools."""
    
    @staticmethod
    def convert_tool(
        mcp_tool: MCPTool,
        mcp_client: MCPClient,
        handle_errors: bool = True
    ) -> BaseTool:
        """Convert a single MCP tool to LangChain tool.
        
        Args:
            mcp_tool: MCP tool definition
            mcp_client: MCP client instance
            handle_errors: Whether to handle tool errors gracefully
            
        Returns:
            LangChain tool instance
        """
        try:
            # Create input model from schema
            input_model = create_pydantic_model_from_schema(
                mcp_tool.name,
                mcp_tool.input_schema,
                mcp_tool.description
            )
            
            # Create tool class dynamically
            tool_class = type(
                f"MCP_{mcp_tool.name}_Tool",
                (MCPBaseTool,),
                {
                    'name': mcp_tool.name,
                    'description': mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                    'args_schema': input_model,
                    'mcp_client': mcp_client,
                    'mcp_tool_name': mcp_tool.name,
                    'handle_tool_error': handle_errors
                }
            )
            
            return tool_class()
            
        except Exception as e:
            raise MCPToolConversionError(mcp_tool.name, str(e))
    
    @staticmethod
    def convert_tools(
        mcp_tools: List[MCPTool],
        mcp_client: MCPClient,
        handle_errors: bool = True,
        filter_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """Convert multiple MCP tools to LangChain tools.
        
        Args:
            mcp_tools: List of MCP tools
            mcp_client: MCP client instance
            handle_errors: Whether to handle tool errors gracefully
            filter_tools: If provided, only convert these tools
            exclude_tools: If provided, exclude these tools
            
        Returns:
            List of LangChain tool instances
        """
        converted_tools = []
        
        for mcp_tool in mcp_tools:
            # Apply filtering
            if filter_tools and mcp_tool.name not in filter_tools:
                continue
            if exclude_tools and mcp_tool.name in exclude_tools:
                continue
            
            try:
                tool = MCPToolConverter.convert_tool(mcp_tool, mcp_client, handle_errors)
                converted_tools.append(tool)
            except MCPToolConversionError as e:
                logger.warning(f"Failed to convert tool '{mcp_tool.name}': {e}")
        
        return converted_tools


class MCPLangChainAdapter:
    """Adapter for integrating MCP with LangChain."""
    
    def __init__(
        self,
        handle_errors: bool = True,
        lazy_loading: bool = True,
        cache_tools: bool = True
    ):
        """Initialize MCP LangChain adapter.
        
        Args:
            handle_errors: Whether to handle tool errors gracefully
            lazy_loading: Whether to use lazy loading for tools
            cache_tools: Whether to cache converted tools
        """
        self.handle_errors = handle_errors
        self.lazy_loading = lazy_loading
        self.cache_tools = cache_tools
        self._tool_cache: Dict[str, Dict[str, BaseTool]] = {}
    
    async def create_tools(
        self,
        mcp_client: MCPClient,
        filter_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """Create LangChain tools from MCP client.
        
        Args:
            mcp_client: MCP client instance
            filter_tools: If provided, only create these tools
            exclude_tools: If provided, exclude these tools
            
        Returns:
            List of LangChain tool instances
        """
        # Ensure client is connected
        if not mcp_client.is_connected:
            await mcp_client.connect()
        
        # Check cache
        client_id = mcp_client.config.name
        if self.cache_tools and client_id in self._tool_cache:
            cached_tools = list(self._tool_cache[client_id].values())
            
            # Apply filtering to cached tools
            if filter_tools:
                cached_tools = [t for t in cached_tools if t.name in filter_tools]
            if exclude_tools:
                cached_tools = [t for t in cached_tools if t.name not in exclude_tools]
            
            return cached_tools
        
        # Convert tools
        if self.lazy_loading:
            tools = self._create_lazy_tools(mcp_client, filter_tools, exclude_tools)
        else:
            tools = MCPToolConverter.convert_tools(
                mcp_client.tools,
                mcp_client,
                self.handle_errors,
                filter_tools,
                exclude_tools
            )
        
        # Cache tools
        if self.cache_tools:
            if client_id not in self._tool_cache:
                self._tool_cache[client_id] = {}
            for tool in tools:
                self._tool_cache[client_id][tool.name] = tool
        
        return tools
    
    def _create_lazy_tools(
        self,
        mcp_client: MCPClient,
        filter_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """Create lazy-loading LangChain tools.
        
        Args:
            mcp_client: MCP client instance
            filter_tools: If provided, only create these tools
            exclude_tools: If provided, exclude these tools
            
        Returns:
            List of lazy-loading LangChain tool instances
        """
        lazy_tools = []
        
        for mcp_tool in mcp_client.tools:
            # Apply filtering
            if filter_tools and mcp_tool.name not in filter_tools:
                continue
            if exclude_tools and mcp_tool.name in exclude_tools:
                continue
            
            # Create lazy tool
            lazy_tool = self._create_lazy_tool(mcp_tool, mcp_client)
            lazy_tools.append(lazy_tool)
        
        return lazy_tools
    
    def _create_lazy_tool(self, mcp_tool: MCPTool, mcp_client: MCPClient) -> BaseTool:
        """Create a single lazy-loading tool.
        
        Args:
            mcp_tool: MCP tool definition
            mcp_client: MCP client instance
            
        Returns:
            Lazy-loading LangChain tool instance
        """
        # Create input model
        input_model = create_pydantic_model_from_schema(
            mcp_tool.name,
            mcp_tool.input_schema,
            mcp_tool.description
        )
        
        class LazyMCPTool(BaseTool):
            """Lazy-loading MCP tool."""
            
            name: str = mcp_tool.name
            description: str = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
            args_schema: Type[BaseModel] = input_model
            handle_tool_error: bool = True
            
            def __init__(self):
                super().__init__()
                self._mcp_client = mcp_client
                self._mcp_tool_name = mcp_tool.name
                self._real_tool: Optional[BaseTool] = None
            
            def _run(self, *args, **kwargs) -> Any:
                raise NotImplementedError("MCP tools only support async operations. Use arun instead.")
            
            async def _arun(
                self,
                *args,
                run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
                **kwargs
            ) -> Any:
                # Load real tool on first use
                if self._real_tool is None:
                    self._real_tool = MCPToolConverter.convert_tool(
                        mcp_tool,
                        self._mcp_client,
                        self.handle_tool_error
                    )
                
                # Execute through real tool
                return await self._real_tool._arun(*args, run_manager=run_manager, **kwargs)
        
        return LazyMCPTool()
    
    def create_structured_tool(
        self,
        mcp_client: MCPClient,
        tool_name: str,
        custom_description: Optional[str] = None,
        custom_schema: Optional[Type[BaseModel]] = None
    ) -> StructuredTool:
        """Create a structured tool from MCP tool.
        
        Args:
            mcp_client: MCP client instance
            tool_name: Name of the MCP tool
            custom_description: Custom description (overrides MCP description)
            custom_schema: Custom input schema (overrides MCP schema)
            
        Returns:
            Structured tool instance
        """
        # Find the MCP tool
        mcp_tool = next((t for t in mcp_client.tools if t.name == tool_name), None)
        if not mcp_tool:
            raise MCPToolConversionError(tool_name, f"Tool '{tool_name}' not found in MCP client")
        
        # Use custom or default values
        description = custom_description or mcp_tool.description or f"MCP tool: {tool_name}"
        
        if custom_schema:
            args_schema = custom_schema
        else:
            args_schema = create_pydantic_model_from_schema(
                mcp_tool.name,
                mcp_tool.input_schema,
                mcp_tool.description
            )
        
        # Create async function wrapper
        async def tool_func(**kwargs) -> str:
            result = await mcp_client.call_tool(tool_name, kwargs)
            
            # Process result
            if isinstance(result, list):
                processed_results = []
                for item in result:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        processed_results.append(item.get('text', ''))
                    else:
                        processed_results.append(str(item))
                return '\n'.join(processed_results)
            else:
                return str(result)
        
        # Create structured tool
        return StructuredTool(
            name=tool_name,
            description=description,
            args_schema=args_schema,
            coroutine=tool_func
        )
    
    def clear_cache(self, client_name: Optional[str] = None):
        """Clear tool cache.
        
        Args:
            client_name: If provided, only clear cache for this client
        """
        if client_name:
            if client_name in self._tool_cache:
                del self._tool_cache[client_name]
        else:
            self._tool_cache.clear()
    
    async def batch_create_tools(
        self,
        mcp_clients: List[MCPClient],
        batch_size: int = 10
    ) -> Dict[str, List[BaseTool]]:
        """Create tools from multiple MCP clients in batches.
        
        Args:
            mcp_clients: List of MCP client instances
            batch_size: Number of clients to process in parallel
            
        Returns:
            Dictionary mapping client names to their tools
        """
        all_tools = {}
        
        # Process clients in batches
        for i in range(0, len(mcp_clients), batch_size):
            batch = mcp_clients[i:i + batch_size]
            
            # Create tools for batch in parallel
            tasks = [self.create_tools(client) for client in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for client, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to create tools for '{client.config.name}': {result}")
                    all_tools[client.config.name] = []
                else:
                    all_tools[client.config.name] = result
        
        return all_tools