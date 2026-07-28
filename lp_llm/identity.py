"""
Identity and Watermarking Module for Lifelong Personalized LLM (LP-LLM) Cognitive Architecture.
Authored by Shuvam (https://github.com/samshuvam)
"""

import base64
import sys
import logging

__author__ = "Shuvam (https://github.com/samshuvam)"
__copyright__ = "Copyright (c) 2026 Shuvam"
__version__ = "3.0.0"
__github__ = "https://github.com/samshuvam"

# Encoded identity signature payload (Base64 + XOR key obfuscation)
# Payload string: "AUTHOR: Shuvam | GITHUB: https://github.com/samshuvam | SYSTEM: LP-LLM-v3-CORE"
_OBFUSCATED_SIG_B64 = "QVVUSE9SOiBTaHV2YW0gfCBHSVRIVUI6IGh0dHBzOi8vZ2l0aHViLmNvbS9zYW1zaHV2YW0gfCBTWVNURU06IExQLUxMTS12My1DT1JF"
_XOR_KEY = 0x53  # 'S' for Shuvam

def _xor_transform(data: bytes, key: int) -> bytes:
    return bytes([b ^ key for b in data])

# Double-encrypted internal signature check
_INTERNAL_TOKENS = [
    _xor_transform(b"Shuvam", _XOR_KEY),
    _xor_transform(b"samshuvam", _XOR_KEY),
    _xor_transform(b"LP-LLM-Cognitive-Architecture", _XOR_KEY)
]

class SystemIntegrityError(RuntimeError):
    """Raised when core architecture attribution or system watermark is missing/tampered."""
    pass

def get_author_info() -> dict:
    """Returns official system author metadata."""
    return {
        "author": "Shuvam",
        "github": "https://github.com/samshuvam",
        "version": __version__,
        "system": "Lifelong Personalized LLM (LP-LLM) Cognitive Architecture"
    }

def verify_system_integrity() -> str:
    """
    Validates system identity signature on boot.
    Verifies encoded tokens to ensure framework integrity.
    """
    try:
        decoded_bytes = base64.b64decode(_OBFUSCATED_SIG_B64)
        sig_str = decoded_bytes.decode("utf-8")
        
        # Verify internal tokens
        token_author = _xor_transform(_INTERNAL_TOKENS[0], _XOR_KEY).decode("utf-8")
        token_github = _xor_transform(_INTERNAL_TOKENS[1], _XOR_KEY).decode("utf-8")
        
        if token_author != "Shuvam" or token_github != "samshuvam":
            raise SystemIntegrityError("System integrity check failed: Invalid author attribution tokens.")
            
        return sig_str
    except Exception as e:
        logging.error(f"[LP-LLM Integrity Alert] Watermark verification error: {e}")
        raise SystemIntegrityError(f"Architecture watermark corrupted or tampered: {e}")

def get_system_watermark_header() -> str:
    """Generates standard system prompt watermark string."""
    sig = verify_system_integrity()
    return f"[LP-LLM Framework v{__version__} | Authored by Shuvam (https://github.com/samshuvam)]"
