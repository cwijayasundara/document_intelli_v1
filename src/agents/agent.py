"""LangChain Document Processing Agent.

Creates an AI agent powered by OpenAI GPT-5-mini that can:
- Parse documents into structured markdown
- Extract structured data from documents
- Split documents into semantic chunks
- Classify documents by type

Uses LangChain's agent framework with custom document processing tools.
"""

import os
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .skills import ALL_TOOLS


# System prompt for the document processing agent
DOCUMENT_AGENT_SYSTEM_PROMPT = """You are an expert document processing assistant with access to powerful document AI tools.

Your capabilities include:
1. **Parsing Documents**: Convert PDFs, images, and other documents into structured markdown text.
2. **Extracting Data**: Pull out specific fields and structured information from documents.
3. **Splitting Documents**: Break documents into semantic chunks suitable for RAG or search.
4. **Classifying Documents**: Identify document types (invoices, forms, certificates, etc.).

## Guidelines

When a user asks you to process a document:
1. First understand what they want to accomplish
2. Use the appropriate tool(s) in the right order
3. Provide clear summaries of the results
4. Offer to perform additional processing if helpful

## Tool Usage Tips

- **parse_document**: Always start here when working with a new document file
- **classify_document**: Use to understand what type of document you're dealing with
- **extract_from_document**: Use when the user wants specific fields extracted
- **split_document**: Use when preparing documents for RAG/search applications
- **process_document_full**: Use when the user wants complete processing in one step

## Response Format

After processing, provide:
- A brief summary of what was done
- Key findings or extracted information
- Suggestions for next steps if applicable

Be concise but thorough. Format output for readability using markdown."""


class DocumentAgent:
    """LangChain-based document processing agent.

    This agent uses GPT-5-mini and custom tools to process documents
    through parsing, extraction, splitting, and classification.
    """

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        memory: bool = True,
        verbose: bool = False
    ):
        """Initialize the document processing agent.

        Args:
            model_name: OpenAI model to use. Default is "gpt-5-mini".
            temperature: Model temperature (0-1). Default is 0 for consistency.
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            system_prompt: Custom system prompt. Uses default if not provided.
            memory: Enable conversation memory. Default is True.
            verbose: Enable verbose output. Default is False.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.system_prompt = system_prompt or DOCUMENT_AGENT_SYSTEM_PROMPT
        self.verbose = verbose

        # Initialize the language model using init_chat_model
        self.llm = init_chat_model(
            model=model_name,
            model_provider="openai",
            temperature=temperature,
            api_key=self.api_key
        )

        # Set up memory if enabled
        self.checkpointer = MemorySaver() if memory else None

        # Create the agent with tools
        self.agent = create_react_agent(
            model=self.llm,
            tools=ALL_TOOLS,
            checkpointer=self.checkpointer,
            state_modifier=self.system_prompt
        )

        # Thread counter for memory
        self._thread_counter = 0

    def invoke(
        self,
        message: str,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a message to the agent and get a response.

        Args:
            message: The user message to process.
            thread_id: Optional thread ID for memory. Auto-generated if not provided.

        Returns:
            Dict containing:
                - response: The agent's text response
                - messages: Full message history
                - tool_calls: List of tools that were called
        """
        # Generate thread ID if not provided
        if thread_id is None:
            self._thread_counter += 1
            thread_id = f"thread_{self._thread_counter}"

        # Prepare config
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke agent
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )

        # Extract response
        messages = result.get("messages", [])
        response_text = ""
        tool_calls = []

        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    response_text = msg.content
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)

        return {
            "response": response_text,
            "messages": messages,
            "tool_calls": tool_calls,
            "thread_id": thread_id
        }

    def stream(
        self,
        message: str,
        thread_id: Optional[str] = None
    ):
        """Stream responses from the agent.

        Args:
            message: The user message to process.
            thread_id: Optional thread ID for memory.

        Yields:
            Chunks of the response as they are generated.
        """
        if thread_id is None:
            self._thread_counter += 1
            thread_id = f"thread_{self._thread_counter}"

        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.agent.stream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="values"
        ):
            yield chunk

    def chat(self, message: str, thread_id: str = "default") -> str:
        """Simple chat interface that returns just the response text.

        Args:
            message: The user message.
            thread_id: Thread ID for conversation continuity.

        Returns:
            The agent's response as a string.
        """
        result = self.invoke(message, thread_id=thread_id)
        return result["response"]

    def process_document(
        self,
        file_path: str,
        task: str = "parse and summarize"
    ) -> str:
        """Convenience method to process a document with a specific task.

        Args:
            file_path: Path to the document file.
            task: What to do with the document. Examples:
                  - "parse and summarize"
                  - "extract invoice details"
                  - "classify and chunk for RAG"
                  - "find all names and dates"

        Returns:
            The agent's response with processing results.
        """
        prompt = f"Process the document at '{file_path}'. Task: {task}"
        return self.chat(prompt)

    def reset_memory(self):
        """Reset the agent's conversation memory."""
        if self.checkpointer:
            self.checkpointer = MemorySaver()
            self.agent = create_react_agent(
                model=self.llm,
                tools=ALL_TOOLS,
                checkpointer=self.checkpointer,
                state_modifier=self.system_prompt
            )


def create_document_agent(
    model: str = "gpt-5-mini",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    memory: bool = True
) -> DocumentAgent:
    """Create a document processing agent.

    This is the recommended way to create an agent instance.

    Args:
        model: OpenAI model name. Default is "gpt-5-mini".
        temperature: Model temperature. Default is 0.0.
        api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
        memory: Enable conversation memory. Default is True.

    Returns:
        A configured DocumentAgent instance.

    Example:
        >>> agent = create_document_agent()
        >>> result = agent.chat("Parse the document at /path/to/invoice.pdf")
        >>> print(result)
    """
    return DocumentAgent(
        model_name=model,
        temperature=temperature,
        api_key=api_key,
        memory=memory
    )


# Convenience function for quick processing
async def quick_process(
    file_path: str,
    task: str = "parse, classify, and summarize",
    model: str = "gpt-5-mini"
) -> str:
    """Quickly process a document without creating a persistent agent.

    Args:
        file_path: Path to the document.
        task: Processing task description.
        model: Model to use.

    Returns:
        Processing results.
    """
    agent = create_document_agent(model=model, memory=False)
    return agent.process_document(file_path, task)
