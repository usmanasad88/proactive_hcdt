"""
Tool registry for managing and accessing AI-callable tools.

The ToolRegistry maintains a collection of available tools and provides
methods for registering, retrieving, and executing them.
"""

from typing import Any

from proactive_hcdt.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """
    Registry for managing AI-callable tools.

    The registry maintains a collection of tools that can be invoked
    by the central AI controller to interact with the robot and environment.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool with the registry.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[BaseTool]) -> None:
        """
        Register multiple tools at once.

        Args:
            tools: List of tool instances to register.
        """
        for tool in tools:
            self.register(tool)

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve.

        Returns:
            The tool instance, or None if not found.
        """
        return self._tools.get(tool_name)

    def has(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is registered, False otherwise.
        """
        return tool_name in self._tools

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Arguments to pass to the tool.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
            )

        # Validate arguments
        valid, error = tool.validate_arguments(**kwargs)
        if not valid:
            return ToolResult(success=False, error=error)

        # Execute the tool
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
            )

    def list_tools(self) -> list[str]:
        """
        Get a list of all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> list[BaseTool]:
        """
        Get all registered tools.

        Returns:
            List of all tool instances.
        """
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """
        Get JSON schemas for all registered tools.

        Returns:
            List of tool schemas in the standard format.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return self.has(tool_name)

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"
