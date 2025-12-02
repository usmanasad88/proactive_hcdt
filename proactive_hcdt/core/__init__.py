"""
Core components of the proactive robotic assistance framework.

This module contains the central AI controller and tool registry
that orchestrate the robot's behavior and capabilities.
"""

from proactive_hcdt.core.controller import AIController
from proactive_hcdt.core.tool_registry import ToolRegistry

__all__ = ["AIController", "ToolRegistry"]
