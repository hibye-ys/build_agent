"""Basic tools for agents."""

import os
import json
import math
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.tools import tool, Tool
from langchain_core.pydantic_v1 import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")


class WeatherInput(BaseModel):
    """Input for weather tool."""
    location: str = Field(description="City name or location")


class FileReadInput(BaseModel):
    """Input for file read tool."""
    path: str = Field(description="Path to the file to read")


class FileWriteInput(BaseModel):
    """Input for file write tool."""
    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class WebSearchInput(BaseModel):
    """Input for web search tool."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results")


@tool("calculator", args_schema=CalculatorInput, return_direct=False)
def calculator(expression: str) -> str:
    """Calculate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        # Use a safe evaluation method
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})
        
        # Remove any potentially dangerous characters
        expression = expression.replace("import", "").replace("exec", "").replace("eval", "")
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool("get_weather", args_schema=WeatherInput, return_direct=False)
def get_weather(location: str) -> str:
    """Get weather information for a location.
    
    Args:
        location: City name or location
        
    Returns:
        Weather information (simulated)
    """
    # This is a mock implementation. In production, you would call a weather API
    weather_data = {
        "Seoul": {"temp": "10°C", "condition": "Partly cloudy", "humidity": "45%"},
        "New York": {"temp": "5°C", "condition": "Clear", "humidity": "60%"},
        "London": {"temp": "8°C", "condition": "Rainy", "humidity": "80%"},
        "Tokyo": {"temp": "12°C", "condition": "Sunny", "humidity": "50%"},
        "San Francisco": {"temp": "15°C", "condition": "Foggy", "humidity": "70%"},
    }
    
    if location in weather_data:
        data = weather_data[location]
        return f"Weather in {location}: {data['condition']}, Temperature: {data['temp']}, Humidity: {data['humidity']}"
    else:
        return f"Weather in {location}: Temperature: 15°C, Condition: Partly cloudy, Humidity: 55% (simulated data)"


@tool("get_datetime", return_direct=False)
def get_datetime() -> str:
    """Get current date and time.
    
    Returns:
        Current datetime string
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@tool("read_file", args_schema=FileReadInput, return_direct=False)
def read_file(path: str) -> str:
    """Read contents of a file.
    
    Args:
        path: Path to the file
        
    Returns:
        File contents or error message
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File {path} does not exist"
        
        if not file_path.is_file():
            return f"Error: {path} is not a file"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return f"File contents of {path}:\n{content}"
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"


@tool("write_file", args_schema=FileWriteInput, return_direct=False)
def write_file(path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        path: Path to the file
        content: Content to write
        
    Returns:
        Success message or error
    """
    try:
        file_path = Path(path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing to file {path}: {str(e)}"


@tool("web_search", args_schema=WebSearchInput, return_direct=False)
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Search results (simulated)
    """
    # This is a mock implementation. In production, you would use a search API
    # like Google Custom Search, Bing Search, or SerpAPI
    
    mock_results = [
        {
            "title": f"Result 1 for '{query}'",
            "url": f"https://example.com/1/{query.replace(' ', '-')}",
            "snippet": f"This is a relevant result about {query}. It contains useful information..."
        },
        {
            "title": f"Wikipedia - {query}",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "snippet": f"Encyclopedia article about {query} with comprehensive information..."
        },
        {
            "title": f"Latest news on {query}",
            "url": f"https://news.example.com/{query.replace(' ', '-')}",
            "snippet": f"Recent developments and updates regarding {query}..."
        }
    ]
    
    results = mock_results[:max_results]
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(
            f"{i}. {result['title']}\n"
            f"   URL: {result['url']}\n"
            f"   {result['snippet']}"
        )
    
    return f"Search results for '{query}':\n" + "\n\n".join(formatted_results)


def get_calculator_tool() -> Tool:
    """Get calculator tool."""
    return calculator


def get_weather_tool() -> Tool:
    """Get weather tool."""
    return get_weather


def get_datetime_tool() -> Tool:
    """Get datetime tool."""
    return get_datetime


def get_file_tools() -> List[Tool]:
    """Get file operation tools."""
    return [read_file, write_file]


def get_web_search_tool() -> Tool:
    """Get web search tool."""
    return web_search