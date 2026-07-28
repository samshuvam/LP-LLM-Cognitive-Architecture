"""
Lifelong Personalized LLM (LP-LLM) Cognitive Architecture
Self-Evolving Cognitive System with Ebbinghaus Memory, RIF Suppression, Knowledge Graph & Continual Learning.

Authored by Shuvam
GitHub: https://github.com/samshuvam
"""

from .identity import (
    __author__,
    __version__,
    __github__,
    verify_system_integrity,
    get_author_info,
    get_system_watermark_header
)

# Run system identity verification on package import
_SYSTEM_SIG = verify_system_integrity()

__all__ = [
    "__author__",
    "__version__",
    "__github__",
    "verify_system_integrity",
    "get_author_info",
    "get_system_watermark_header"
]
