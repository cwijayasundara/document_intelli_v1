"""Demo script for the Document Processing Agent.

This script demonstrates how to use the LangChain-based document
processing agent with its various capabilities.

Usage:
    python -m src.agents.demo

Or with a specific document:
    python -m src.agents.demo /path/to/document.pdf
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


def check_api_keys():
    """Check required API keys are set."""
    keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "LLAMA_CLOUD_API_KEY": os.environ.get("LLAMA_CLOUD_API_KEY"),
    }

    missing = [k for k, v in keys.items() if not v]
    if missing:
        print("Missing required API keys:")
        for key in missing:
            print(f"  - {key}")
        print("\nPlease set these in your .env file or environment.")
        return False
    return True


def demo_basic_usage():
    """Demonstrate basic agent usage."""
    print("\n" + "=" * 60)
    print("BASIC AGENT USAGE DEMO")
    print("=" * 60)

    from src.agents import create_document_agent

    # Create agent
    print("\nCreating document agent with GPT-5-mini...")
    agent = create_document_agent(
        model="gpt-5-mini",
        temperature=0.0
    )
    print("Agent created successfully!")

    # Simple conversation
    print("\n--- Simple Conversation ---")
    response = agent.chat(
        "What document processing capabilities do you have?",
        thread_id="demo"
    )
    print(f"Agent: {response}")

    return agent


def demo_document_processing(agent, file_path: str):
    """Demonstrate document processing."""
    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSING DEMO")
    print("=" * 60)

    print(f"\nProcessing: {file_path}")

    # Parse the document
    print("\n--- Parsing Document ---")
    response = agent.chat(
        f"Parse the document at '{file_path}' and give me a summary of its contents.",
        thread_id="demo"
    )
    print(f"Agent: {response[:1000]}...")

    # Classify the document
    print("\n--- Classifying Document ---")
    response = agent.chat(
        "Now classify this document. What type of document is it?",
        thread_id="demo"
    )
    print(f"Agent: {response}")

    # Extract information
    print("\n--- Extracting Information ---")
    response = agent.chat(
        "Extract any key information like names, dates, and numbers from this document.",
        thread_id="demo"
    )
    print(f"Agent: {response}")


def demo_full_pipeline(file_path: str):
    """Demonstrate the full processing pipeline."""
    print("\n" + "=" * 60)
    print("FULL PIPELINE DEMO")
    print("=" * 60)

    from src.agents import create_document_agent

    agent = create_document_agent()

    print(f"\nProcessing: {file_path}")
    print("Running full pipeline (parse, classify, extract, chunk)...")

    response = agent.chat(
        f"""Process the document at '{file_path}' through the complete pipeline:
        1. Parse it to markdown
        2. Classify the document type
        3. Extract key fields (names, dates, numbers, etc.)
        4. Split it into chunks for RAG

        Provide a comprehensive summary of the results.""",
        thread_id="pipeline_demo"
    )

    print(f"\nAgent Response:\n{response}")


def demo_tools_directly():
    """Demonstrate using tools directly without the agent."""
    print("\n" + "=" * 60)
    print("DIRECT TOOLS USAGE DEMO")
    print("=" * 60)

    from src.agents.skills import (
        parse_document,
        classify_document,
        extract_from_document,
        split_document
    )

    # Get a test document
    test_docs = list(Path("difficult_examples").glob("*.pdf"))
    if not test_docs:
        print("No test documents found in difficult_examples/")
        return

    file_path = str(test_docs[0])
    print(f"\nUsing document: {file_path}")

    # Parse
    print("\n--- Direct Parse ---")
    markdown = parse_document.invoke({
        "file_path": file_path,
        "processor": "llamaindex"
    })
    print(f"Parsed {len(markdown)} characters")
    print(f"Preview: {markdown[:300]}...")

    # Classify
    print("\n--- Direct Classify ---")
    classification = classify_document.invoke({"content": markdown})
    print(f"Classification: {classification}")

    # Extract
    print("\n--- Direct Extract ---")
    extraction = extract_from_document.invoke({
        "content": markdown,
        "fields": "date, name, number, title"
    })
    print(f"Extraction: {extraction}")

    # Split
    print("\n--- Direct Split ---")
    chunks = split_document.invoke({
        "content": markdown,
        "max_chunk_size": 1000
    })
    import json
    chunks_data = json.loads(chunks)
    print(f"Created {chunks_data['total_chunks']} chunks")


def interactive_mode(agent):
    """Run interactive chat with the agent."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Chat with the document processing agent.")
    print("Commands:")
    print("  /quit - Exit interactive mode")
    print("  /reset - Reset conversation memory")
    print("  /help - Show available commands")
    print("-" * 60)

    thread_id = "interactive"

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/reset":
                agent.reset_memory()
                print("Memory reset.")
                continue

            if user_input.lower() == "/help":
                print("""
Available commands:
  /quit  - Exit interactive mode
  /reset - Reset conversation memory
  /help  - Show this help message

Example queries:
  - "Parse the document at difficult_examples/invoice.pdf"
  - "What type of document is this?"
  - "Extract the total amount and date"
  - "Split this into chunks for RAG"
                """)
                continue

            response = agent.chat(user_input, thread_id=thread_id)
            print(f"\nAgent: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main demo entry point."""
    print("=" * 60)
    print("DOCUMENT PROCESSING AGENT DEMO")
    print("=" * 60)

    # Check API keys
    if not check_api_keys():
        return

    # Check for command line argument
    file_path = None
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            return

    try:
        # Basic usage demo
        agent = demo_basic_usage()

        # Document processing demo
        if file_path:
            demo_document_processing(agent, file_path)
        else:
            # Use a test document if available
            test_docs = list(Path("difficult_examples").glob("*.pdf"))
            if test_docs:
                demo_document_processing(agent, str(test_docs[0]))
            else:
                print("\nNo test documents found. Skipping document processing demo.")

        # Ask if user wants interactive mode
        print("\n" + "-" * 60)
        response = input("Enter interactive mode? (y/n): ").strip().lower()
        if response == 'y':
            interactive_mode(agent)

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
