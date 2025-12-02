"""
Tool system for AI-callable robot capabilities.

This module provides the base classes and utilities for creating
tools that can be invoked by the central AI controller.
"""

from proactive_hcdt.tools.base import BaseTool, ToolParameter, ToolResult
from proactive_hcdt.tools.examples import (
    CommunicationTool,
    ManipulationTool,
    MovementTool,
    PerceptionTool,
)

__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    "MovementTool",
    "PerceptionTool",
    "CommunicationTool",
    "ManipulationTool",
]
