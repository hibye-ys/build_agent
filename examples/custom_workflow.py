"""Custom workflow example using WorkflowBuilder."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows import WorkflowBuilder, WorkflowConfig
from src.workflows.templates import create_conditional_workflow


def main():
    """Run custom workflow examples."""
    
    print("🔄 Custom Workflow Examples")
    print("=" * 50)
    
    # Example 1: Simple Sequential Workflow
    print("\n📌 Example 1: Sequential Workflow")
    print("-" * 30)
    
    config = WorkflowConfig(
        name="data_processing_pipeline",
        description="Process data through multiple stages",
        enable_memory=True,
        visualize=True
    )
    
    builder = WorkflowBuilder(config)
    
    # Define workflow nodes
    def fetch_data(state):
        """Fetch data from source."""
        print("  → Fetching data...")
        state["metadata"]["data_fetched"] = True
        state["messages"].append({"content": "Data fetched successfully"})
        return state
    
    def process_data(state):
        """Process the fetched data."""
        print("  → Processing data...")
        if state["metadata"].get("data_fetched"):
            state["metadata"]["data_processed"] = True
            state["messages"].append({"content": "Data processed"})
        return state
    
    def analyze_data(state):
        """Analyze the processed data."""
        print("  → Analyzing data...")
        if state["metadata"].get("data_processed"):
            state["metadata"]["analysis_complete"] = True
            state["messages"].append({"content": "Analysis complete"})
        return state
    
    def generate_report(state):
        """Generate final report."""
        print("  → Generating report...")
        if state["metadata"].get("analysis_complete"):
            state["messages"].append({"content": "Report generated successfully!"})
        return state
    
    # Build the workflow
    workflow = (builder
        .add_node("fetch", fetch_data, "Fetch data from source")
        .add_node("process", process_data, "Process raw data")
        .add_node("analyze", analyze_data, "Analyze processed data")
        .add_node("report", generate_report, "Generate final report")
        .add_edge("START", "fetch")
        .add_edge("fetch", "process")
        .add_edge("process", "analyze")
        .add_edge("analyze", "report")
        .add_edge("report", "END")
        .compile()
    )
    
    # Execute the workflow
    initial_state = {
        "messages": [],
        "current_node": "start",
        "history": [],
        "metadata": {},
        "error": None
    }
    
    try:
        result = workflow.invoke(initial_state)
        print("\n✅ Workflow completed successfully!")
        print(f"Final messages: {len(result['messages'])} messages")
        for msg in result["messages"]:
            print(f"  - {msg['content']}")
    except Exception as e:
        print(f"❌ Workflow error: {e}")
    
    # Example 2: Conditional Workflow
    print("\n📌 Example 2: Conditional Workflow")
    print("-" * 30)
    
    def initial_check(state):
        """Initial data check."""
        print("  → Performing initial check...")
        import random
        # Simulate a condition
        state["metadata"]["data_quality"] = random.choice(["good", "bad"])
        print(f"    Data quality: {state['metadata']['data_quality']}")
        return state
    
    def condition_checker(state):
        """Check data quality condition."""
        quality = state["metadata"].get("data_quality", "unknown")
        return "good_path" if quality == "good" else "bad_path"
    
    def process_good_data(state):
        """Process good quality data."""
        print("  → Processing good quality data...")
        state["messages"].append({"content": "Good data processed successfully"})
        return state
    
    def handle_bad_data(state):
        """Handle bad quality data."""
        print("  → Handling bad quality data...")
        state["messages"].append({"content": "Bad data handled with cleanup"})
        return state
    
    # Create conditional workflow
    conditional_workflow = create_conditional_workflow(
        initial_step={"name": "check", "func": initial_check},
        condition_func=condition_checker,
        branches={
            "good_path": [{"name": "process_good", "func": process_good_data}],
            "bad_path": [{"name": "handle_bad", "func": handle_bad_data}]
        },
        name="quality_based_workflow"
    )
    
    # Execute conditional workflow
    try:
        result = conditional_workflow.invoke(initial_state)
        print("\n✅ Conditional workflow completed!")
        for msg in result.get("messages", []):
            print(f"  - {msg['content']}")
    except Exception as e:
        print(f"❌ Conditional workflow error: {e}")
    
    # Example 3: Workflow with Error Handling
    print("\n📌 Example 3: Workflow with Error Handling")
    print("-" * 30)
    
    error_config = WorkflowConfig(
        name="error_handling_workflow",
        description="Workflow with retry logic",
        enable_memory=True
    )
    
    error_builder = WorkflowBuilder(error_config)
    
    def risky_operation(state):
        """Operation that might fail."""
        import random
        print("  → Attempting risky operation...")
        if random.random() > 0.5:
            raise Exception("Random failure occurred!")
        state["messages"].append({"content": "Operation succeeded"})
        return state
    
    def fallback_operation(state):
        """Fallback when main operation fails."""
        print("  → Executing fallback operation...")
        state["messages"].append({"content": "Fallback executed successfully"})
        return state
    
    # Build workflow with error handling
    error_workflow = (error_builder
        .add_node("risky", risky_operation, "Risky operation", retry_on_error=True, max_retries=2)
        .add_node("fallback", fallback_operation, "Fallback operation")
        .add_edge("START", "risky")
        .add_conditional_edge(
            "risky",
            lambda s: "fallback" if s.get("error") else "END",
            {"fallback": "fallback", "END": "END"}
        )
        .add_edge("fallback", "END")
        .compile()
    )
    
    # Execute error handling workflow
    try:
        result = error_workflow.invoke(initial_state)
        print("\n✅ Error handling workflow completed!")
        for msg in result.get("messages", []):
            print(f"  - {msg['content']}")
        if result.get("error"):
            print(f"  ⚠️ Error recorded: {result['error']}")
    except Exception as e:
        print(f"❌ Error handling workflow failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ All workflow examples completed!")


if __name__ == "__main__":
    main()