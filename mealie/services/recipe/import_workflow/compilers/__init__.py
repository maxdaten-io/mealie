from .base import COMPILE_SOURCE_PROMPT, SourceCompiler, SourceType
from .gemini_video import GeminiVideoCompiler
from .image import ImageCompiler
from .structured_data import StructuredDataCompiler
from .transcription import TranscriptionCompiler
from .web_page import WebPageCompiler

DEFAULT_SOURCE_COMPILERS: list[type[SourceCompiler]] = [
    ImageCompiler,
    GeminiVideoCompiler,
    TranscriptionCompiler,
    StructuredDataCompiler,
    WebPageCompiler,
]
"""Order matters: within a source type, the first compiler that can handle the input is used"""

__all__ = [
    "COMPILE_SOURCE_PROMPT",
    "DEFAULT_SOURCE_COMPILERS",
    "GeminiVideoCompiler",
    "ImageCompiler",
    "SourceCompiler",
    "SourceType",
    "StructuredDataCompiler",
    "TranscriptionCompiler",
    "WebPageCompiler",
]
