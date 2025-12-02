"""
AI Provider implementations for different frontier AI models.

Supports:
- Gemini (Google)
- Claude (Anthropic)
- OpenAI (GPT models)
- Dummy provider for testing
"""

from proactive_hcdt.ai_providers.base import AIMessage, AIProvider, AIResponse
from proactive_hcdt.ai_providers.dummy import DummyAIProvider

__all__ = [
    "AIProvider",
    "AIMessage",
    "AIResponse",
    "DummyAIProvider",
]
