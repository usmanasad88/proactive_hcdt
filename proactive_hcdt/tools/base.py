"""
Base tool template for creating AI-callable tools.

This module provides the abstract base class and utilities for defining
tools that can be called by the central AI controller. All custom tools
should inherit from BaseTool and implement the required methods.

Example:
    ```python
    from proactive_hcdt.tools.base import BaseTool, ToolParameter, ToolResult

    class MyCustomTool(BaseTool):
        name = "my_custom_tool"
        description = "A custom tool that does something useful"
        parameters = [
            ToolParameter(
                name="input_data",
                type="string",
                description="The input data to process",
                required=True
            ),
            ToolParameter(
                name="option",
                type="boolean",
                description="An optional flag",
                required=False,
                default=False
            )
        ]

        async def execute(self, **kwargs) -> ToolResult:
            input_data = kwargs.get("input_data")
            option = kwargs.get("option", False)

            # Do something with the input
            result = f"Processed: {input_data}"

            return ToolResult(success=True, data={"result": result})
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolParameterType(str, Enum):
    """Supported parameter types for tools."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.

    Attributes:
        name: The parameter name (used as the key in kwargs).
        type: The parameter type (string, number, integer, boolean, array, object).
        description: Human-readable description of the parameter.
        required: Whether this parameter is required.
        default: Default value if not provided (only for optional parameters).
        enum: Optional list of allowed values.
        items: For array types, the type of items in the array.
        properties: For object types, the nested property definitions.
    """

    name: str
    type: ToolParameterType | str
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert the parameter to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type.value if isinstance(self.type, ToolParameterType) else self.type,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        if self.items:
            schema["items"] = self.items

        if self.properties:
            schema["properties"] = self.properties

        if self.default is not None:
            schema["default"] = self.default

        return schema


@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Attributes:
        success: Whether the tool execution was successful.
        data: The result data (can be any JSON-serializable type).
        error: Error message if the execution failed.
        metadata: Additional metadata about the execution.
    """

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> str:
        """Convert the result to a message string for the AI."""
        if self.success:
            if isinstance(self.data, dict):
                import json

                return json.dumps(self.data)
            return str(self.data) if self.data is not None else "Success"
        else:
            return f"Error: {self.error}"


class BaseTool(ABC):
    """
    Abstract base class for AI-callable tools.

    All tools in the proactive robotic assistance framework should inherit
    from this class and implement the required attributes and methods.

    Class Attributes:
        name: Unique identifier for the tool (snake_case recommended).
        description: Human-readable description of what the tool does.
        parameters: List of ToolParameter objects defining the tool's inputs.

    Example Implementation:
        ```python
        class GreetingTool(BaseTool):
            name = "greet_person"
            description = "Greet a person by name with an optional custom message"
            parameters = [
                ToolParameter(
                    name="person_name",
                    type="string",
                    description="Name of the person to greet",
                    required=True
                ),
                ToolParameter(
                    name="custom_greeting",
                    type="string",
                    description="Custom greeting message",
                    required=False,
                    default="Hello"
                )
            ]

            async def execute(self, **kwargs) -> ToolResult:
                name = kwargs.get("person_name")
                greeting = kwargs.get("custom_greeting", "Hello")
                return ToolResult(
                    success=True,
                    data={"message": f"{greeting}, {name}!"}
                )
        ```
    """

    # Required class attributes - subclasses must define these
    name: str
    description: str
    parameters: list[ToolParameter] = []

    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses define required attributes."""
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract classes
        if ABC in cls.__bases__:
            return

        if not hasattr(cls, "name") or not cls.name:
            raise TypeError(f"Tool class {cls.__name__} must define 'name' attribute")

        if not hasattr(cls, "description") or not cls.description:
            raise TypeError(f"Tool class {cls.__name__} must define 'description' attribute")

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Tool arguments as keyword arguments.

        Returns:
            ToolResult containing the execution outcome.

        Note:
            Implementations should handle errors gracefully and return
            ToolResult with success=False and an error message rather
            than raising exceptions.
        """
        pass

    def validate_arguments(self, **kwargs: Any) -> tuple[bool, str | None]:
        """
        Validate that the provided arguments match the parameter definitions.

        Args:
            **kwargs: Arguments to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return False, f"Missing required parameter: {param.name}"

            if param.name in kwargs:
                value = kwargs[param.name]
                if not self._validate_type(value, param.type):
                    return False, f"Invalid type for parameter {param.name}: expected {param.type}"

                if param.enum and value not in param.enum:
                    return False, f"Invalid value for parameter {param.name}: must be one of {param.enum}"

        return True, None

    def _validate_type(self, value: Any, expected_type: ToolParameterType | str) -> bool:
        """Validate that a value matches the expected type."""
        type_str = expected_type.value if isinstance(expected_type, ToolParameterType) else expected_type

        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_mapping.get(type_str)
        if expected_python_type is None:
            return True  # Unknown types pass validation

        return isinstance(value, expected_python_type)

    def to_schema(self) -> dict[str, Any]:
        """
        Convert the tool to a JSON Schema format suitable for AI providers.

        Returns:
            Dictionary in the standard tool schema format.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }

        if required:
            schema["parameters"]["required"] = required

        return schema

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
