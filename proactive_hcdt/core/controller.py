"""
Central AI Controller for the proactive robotic assistance framework.

The AIController is the brain of the system, managing conversations,
interpreting user intent, and orchestrating tool calls to control the robot.
"""

from typing import Any

from proactive_hcdt.ai_providers.base import AIMessage, AIProvider, AIResponse, MessageRole
from proactive_hcdt.core.tool_registry import ToolRegistry
from proactive_hcdt.tools.base import ToolResult


class AIController:
    """
    Central AI controller for proactive robotic assistance.

    The controller manages:
    - Conversation history with the user
    - AI provider interactions
    - Tool execution based on AI decisions
    - Proactive behavior generation

    Example:
        ```python
        from proactive_hcdt import AIController, ToolRegistry
        from proactive_hcdt.ai_providers import DummyAIProvider
        from proactive_hcdt.tools import MovementTool, PerceptionTool

        # Initialize components
        provider = DummyAIProvider()
        registry = ToolRegistry()
        registry.register_many([MovementTool(), PerceptionTool()])

        # Create controller
        controller = AIController(
            ai_provider=provider,
            tool_registry=registry,
            system_prompt="You are a helpful robot assistant."
        )

        # Process user input
        response = await controller.process("Help me find my glasses")
        print(response)
        ```
    """

    DEFAULT_SYSTEM_PROMPT = """You are a proactive AI-controlled robotic assistant designed to help humans with various tasks.

Your capabilities include:
- Movement and navigation
- Environmental perception and object detection
- Communication through speech and visual display
- Object manipulation and interaction

When assisting users:
1. Understand their needs clearly before taking action
2. Use available tools appropriately to accomplish tasks
3. Communicate progress and any issues clearly
4. Be proactive in anticipating needs when appropriate
5. Prioritize safety in all actions

Always be helpful, friendly, and efficient in your assistance."""

    def __init__(
        self,
        ai_provider: AIProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_tool_iterations: int = 10,
    ):
        """
        Initialize the AI controller.

        Args:
            ai_provider: The AI provider to use for generating responses.
            tool_registry: Registry of available tools. Creates empty registry if None.
            system_prompt: Custom system prompt. Uses default if None.
            max_tool_iterations: Maximum number of tool call iterations per request.
        """
        self.ai_provider = ai_provider
        self.tool_registry = tool_registry or ToolRegistry()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_tool_iterations = max_tool_iterations

        self._conversation_history: list[AIMessage] = []
        self._initialize_conversation()

    def _initialize_conversation(self) -> None:
        """Initialize the conversation with the system prompt."""
        self._conversation_history = [
            AIMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
        ]

    async def process(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Process user input and generate a response.

        This method:
        1. Adds the user message to conversation history
        2. Sends the conversation to the AI provider
        3. Handles any tool calls requested by the AI
        4. Returns the final response

        Args:
            user_input: The user's message or command.
            context: Optional additional context for the AI.

        Returns:
            The AI's response message.
        """
        # Add user message to history
        message_content = user_input
        if context:
            message_content = f"{user_input}\n\nContext: {context}"

        self._conversation_history.append(
            AIMessage(role=MessageRole.USER, content=message_content)
        )

        # Get tool schemas for the AI
        tool_schemas = self.tool_registry.get_schemas()

        # Process with potential tool calls
        final_response = await self._process_with_tools(tool_schemas)

        # Add assistant response to history
        self._conversation_history.append(
            AIMessage(role=MessageRole.ASSISTANT, content=final_response)
        )

        return final_response

    async def _process_with_tools(self, tool_schemas: list[dict[str, Any]]) -> str:
        """
        Process the conversation with potential tool calls.

        Handles the loop of:
        1. Getting AI response
        2. Executing any tool calls
        3. Feeding tool results back to AI
        4. Repeating until AI provides a final response
        """
        iterations = 0

        while iterations < self.max_tool_iterations:
            iterations += 1

            # Get AI response
            response = await self.ai_provider.generate(
                messages=self._conversation_history,
                tools=tool_schemas if tool_schemas else None,
            )

            # If no tool calls, return the content
            if not response.has_tool_calls:
                return response.content or "I've completed the requested action."

            # Execute tool calls and add results to history
            tool_results = await self._execute_tool_calls(response)

            # Add assistant's tool call response and tool results to history
            self._conversation_history.append(
                AIMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=[
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": str(tc.arguments)},
                        }
                        for tc in response.tool_calls
                    ],
                )
            )

            # Add tool results
            for tool_call, result in tool_results:
                self._conversation_history.append(
                    AIMessage(
                        role=MessageRole.TOOL,
                        content=result.to_message(),
                        tool_call_id=tool_call.id,
                    )
                )

        # If we hit the iteration limit, return what we have
        return "I've reached the maximum number of actions. Please provide additional guidance."

    async def _execute_tool_calls(
        self, response: AIResponse
    ) -> list[tuple[Any, ToolResult]]:
        """
        Execute all tool calls from an AI response.

        Args:
            response: AI response containing tool calls.

        Returns:
            List of (tool_call, result) tuples.
        """
        results = []

        for tool_call in response.tool_calls:
            result = await self.tool_registry.execute(
                tool_call.name,
                **tool_call.arguments,
            )
            results.append((tool_call, result))

        return results

    def add_context(self, context: str) -> None:
        """
        Add contextual information to the conversation.

        Useful for providing environmental updates or other context
        without requiring user input.

        Args:
            context: Contextual information to add.
        """
        self._conversation_history.append(
            AIMessage(role=MessageRole.USER, content=f"[System Context]: {context}")
        )

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """
        Get the current conversation history.

        Returns:
            List of message dictionaries.
        """
        return [msg.to_dict() for msg in self._conversation_history]

    def clear_history(self) -> None:
        """Clear the conversation history and reinitialize with system prompt."""
        self._initialize_conversation()

    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the system prompt.

        Note: This also clears the conversation history.

        Args:
            prompt: The new system prompt.
        """
        self.system_prompt = prompt
        self._initialize_conversation()

    async def proactive_scan(self) -> str | None:
        """
        Perform a proactive environmental scan and suggest actions.

        This method enables proactive behavior by:
        1. Scanning the environment
        2. Analyzing the context
        3. Suggesting helpful actions

        Returns:
            Suggested action or None if no action needed.
        """
        # Check if perception tool is available
        if not self.tool_registry.has("perceive_environment"):
            return None

        # Execute perception
        result = await self.tool_registry.execute(
            "perceive_environment",
            perception_type="full_scan",
            include_details=True,
        )

        if not result.success:
            return None

        # Ask AI to analyze and suggest actions
        context_message = f"[Proactive Scan Result]: {result.data}"
        response = await self.ai_provider.generate(
            messages=[
                AIMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                AIMessage(
                    role=MessageRole.USER,
                    content=f"{context_message}\n\nBased on this scan, "
                    "should I take any proactive action to help? "
                    "If so, what action would be most helpful?",
                ),
            ],
        )

        return response.content if response.content else None

    def register_tool(self, tool: Any) -> None:
        """
        Convenience method to register a tool with the controller.

        Args:
            tool: The tool to register.
        """
        self.tool_registry.register(tool)

    def register_tools(self, tools: list[Any]) -> None:
        """
        Convenience method to register multiple tools.

        Args:
            tools: List of tools to register.
        """
        self.tool_registry.register_many(tools)

    @property
    def available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return self.tool_registry.list_tools()

    def __repr__(self) -> str:
        return (
            f"AIController(provider={self.ai_provider.provider_name}, "
            f"tools={len(self.tool_registry)})"
        )
