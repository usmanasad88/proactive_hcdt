"""
Proactive AI-Controlled Robotic Assistance Framework for Human-Robot Interaction.

This framework provides a modular architecture for building proactive AI-controlled
robotic assistants that can interact with humans naturally and anticipate their needs.

The architecture supports multiple AI providers (Gemini, Claude, OpenAI) and includes
a flexible tool system that allows the central AI to invoke various robot capabilities.
"""

from proactive_hcdt.core.controller import AIController
from proactive_hcdt.core.tool_registry import ToolRegistry
from proactive_hcdt.tools.base import BaseTool, ToolParameter, ToolResult

__version__ = "0.1.0"

__all__ = [
    "AIController",
    "ToolRegistry",
    "BaseTool",
    "ToolParameter",
    "ToolResult",
]
