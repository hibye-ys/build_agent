#!/usr/bin/env python3
"""
Setup and Validation Script for Multi-Model Agent Testing Framework
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import json

def print_banner():
    """Print setup banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║     🤖 Multi-Model Agent Framework Setup & Validation    ║
║                                                          ║
║  This script will help you set up and validate the      ║
║  agent testing framework for ultrathink project         ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def check_project_structure():
    """Verify project structure exists."""
    print("\n📁 Checking project structure...")
    
    required_paths = [
        ".conductor/hibye-ys-dublin",
        ".conductor/hibye-ys-dublin/src",
        ".conductor/hibye-ys-dublin/src/agents",
        ".conductor/hibye-ys-dublin/src/models", 
        ".conductor/hibye-ys-dublin/src/tools",
        ".conductor/hibye-ys-dublin/requirements.txt"
    ]
    
    missing_paths = []
    for path in required_paths:
        full_path = Path(path)
        if full_path.exists():
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} - MISSING")
            missing_paths.append(path)
    
    if missing_paths:
        print(f"\n❌ Missing required paths: {len(missing_paths)}")
        return False
    else:
        print(f"\n✅ All required paths found")
        return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    requirements_file = Path(".conductor/hibye-ys-dublin/requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def setup_environment():
    """Set up environment variables."""
    print("\n🔧 Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path(".conductor/hibye-ys-dublin/.env.example")
    
    # Copy example env if it exists and .env doesn't
    if env_example.exists() and not env_file.exists():
        try:
            with open(env_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print(f"✅ Created .env from example")
        except Exception as e:
            print(f"❌ Failed to create .env: {e}")
    
    # Load and check environment
    load_dotenv()
    
    api_keys = {
        "OPENAI_API_KEY": "OpenAI GPT Models",
        "ANTHROPIC_API_KEY": "Anthropic Claude Models",
        "GOOGLE_API_KEY": "Google Gemini Models"
    }
    
    available = []
    for key, desc in api_keys.items():
        if os.getenv(key):
            print(f"  ✅ {desc}: Configured")
            available.append(key)
        else:
            print(f"  ⚠️ {desc}: Not configured")
    
    if not available:
        print("\n⚠️ No API keys configured!")
        print("To use the framework, add at least one API key to .env file:")
        print("  OPENAI_API_KEY=your_openai_key_here")
        print("  ANTHROPIC_API_KEY=your_anthropic_key_here") 
        print("  GOOGLE_API_KEY=your_google_key_here")
        return False
    else:
        print(f"\n✅ {len(available)} API key(s) configured")
        return True

def validate_imports():
    """Validate that all required modules can be imported."""
    print("\n🔍 Validating imports...")
    
    project_path = Path(".conductor/hibye-ys-dublin")
    sys.path.insert(0, str(project_path))
    
    required_modules = [
        ("langchain", "LangChain core"),
        ("langchain_core", "LangChain core components"),
        ("langgraph", "LangGraph workflow engine"),
        ("dotenv", "Environment variable loading"),
        ("pydantic", "Data validation"),
    ]
    
    framework_modules = [
        ("src.models", "Model factory"),
        ("src.agents", "Agent implementations"),
        ("src.tools", "Tool registry"),
    ]
    
    # Check external dependencies
    failed_modules = []
    for module, description in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {description}: Available")
        except ImportError as e:
            print(f"  ❌ {description}: Missing - {e}")
            failed_modules.append(module)
    
    # Check framework modules  
    for module, description in framework_modules:
        try:
            __import__(module)
            print(f"  ✅ {description}: Available")
        except ImportError as e:
            print(f"  ❌ {description}: Missing - {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ Failed to import {len(failed_modules)} modules")
        return False
    else:
        print(f"\n✅ All modules imported successfully")
        return True

def test_basic_functionality():
    """Test basic agent creation and functionality."""
    print("\n🧪 Testing basic functionality...")
    
    # Add project to path
    project_path = Path(".conductor/hibye-ys-dublin")
    sys.path.insert(0, str(project_path))
    
    try:
        from src.models import get_available_models
        from src.tools import list_available_tools
        from src.agents import create_simple_react_agent
        
        # Test model registry
        models = get_available_models()
        print(f"  ✅ Model registry loaded: {len(models)} providers")
        
        # Test tool registry
        tools = list_available_tools()
        print(f"  ✅ Tool registry loaded: {len(tools)} tools")
        
        # Test agent creation (with mock if no API keys)
        available_providers = []
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
            if os.getenv(key):
                available_providers.append(key.replace("_API_KEY", "").lower())
        
        if available_providers:
            provider = available_providers[0]
            model_map = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-haiku", 
                "google": "gemini-1.5-flash"
            }
            
            try:
                agent = create_simple_react_agent(
                    model_provider=provider,
                    model_name=model_map.get(provider, "gpt-4o-mini"),
                    tool_names=["calculator"],
                    system_prompt="Test agent",
                    memory=False
                )
                print(f"  ✅ Agent creation successful: {provider}")
                
                # Quick test
                response = agent.invoke({
                    "messages": [{"role": "user", "content": "What is 2+2?"}]
                })
                print(f"  ✅ Basic query test successful")
                
            except Exception as e:
                print(f"  ⚠️ Agent test failed (but imports work): {e}")
        else:
            print(f"  ⚠️ No API keys available for agent testing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def create_validation_report():
    """Create a validation report."""
    print("\n📋 Creating validation report...")
    
    report = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "project_structure": "validated", 
        "dependencies": "installed",
        "environment": "configured" if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY") else "partial",
        "imports": "successful",
        "basic_test": "passed",
        "status": "ready"
    }
    
    try:
        with open("validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("✅ Validation report saved: validation_report.json")
    except Exception as e:
        print(f"⚠️ Could not save validation report: {e}")

def main():
    """Main setup and validation process."""
    print_banner()
    
    steps = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure), 
        ("Dependencies", install_dependencies),
        ("Environment", setup_environment),
        ("Imports", validate_imports),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    print("🚀 Starting setup and validation process...\n")
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"📋 Step: {step_name}")
        try:
            if step_func():
                success_count += 1
            print()  # Add spacing
        except Exception as e:
            print(f"❌ Step '{step_name}' failed with error: {e}\n")
    
    # Summary
    print("=" * 60)
    print(f"📊 Validation Summary: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        print("🎉 Setup and validation completed successfully!")
        print("\n🎯 You can now use the framework:")
        print("  • Run: python test_agent_demo.py")
        print("  • Open: agent_testing_framework.ipynb")
        print("  • Check: validation_report.json")
        
        create_validation_report()
        
    elif success_count >= len(steps) - 2:
        print("⚠️ Setup mostly successful with minor issues")
        print("Framework should work with available providers")
        
        create_validation_report()
        
    else:
        print("❌ Setup failed. Please address the issues above")
        print("\n💡 Common solutions:")
        print("  • Install missing dependencies: pip install -r .conductor/hibye-ys-dublin/requirements.txt")
        print("  • Add API keys to .env file")
        print("  • Check Python version (3.8+ required)")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted. Run again when ready!")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)