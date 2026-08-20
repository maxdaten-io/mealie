from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mealie.core.config import get_app_settings
from mealie.services.recipe.import_workflow.compilers import (
    DEFAULT_SOURCE_COMPILERS,
    GeminiVideoCompiler,
    TranscriptionCompiler,
)
from mealie.services.recipe.import_workflow.context import WorkflowContext, WorkflowInput, WorkflowOptions


def make_ctx(url: str | None) -> WorkflowContext:
    return WorkflowContext(
        input=WorkflowInput(url=url),
        options=WorkflowOptions(),
        repos=MagicMock(),
        translator=MagicMock(),
        ai=MagicMock(),
    )


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def gemini_enabled(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "True")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_app_settings.cache_clear()


def test_registered_before_transcription_compiler():
    assert GeminiVideoCompiler in DEFAULT_SOURCE_COMPILERS
    assert DEFAULT_SOURCE_COMPILERS.index(GeminiVideoCompiler) < DEFAULT_SOURCE_COMPILERS.index(TranscriptionCompiler)


def test_can_compile_false_when_gemini_disabled():
    compiler = GeminiVideoCompiler(make_ctx("https://www.youtube.com/watch?v=abc123DEF45"))
    assert compiler.can_compile() is False


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc123DEF45",
        "https://youtube.com/watch?v=abc123DEF45",
        "https://m.youtube.com/watch?v=abc123DEF45&t=42",
        "https://www.youtube.com/shorts/abc123DEF45",
        "https://youtu.be/abc123DEF45",
    ],
)
def test_can_compile_youtube_urls(gemini_enabled, url):
    compiler = GeminiVideoCompiler(make_ctx(url))
    assert compiler.can_compile() is True


@pytest.mark.parametrize(
    "url",
    [
        None,
        "https://vimeo.com/123456789",
        "https://example.com/watch?v=abc123DEF45",
        "https://www.youtube.com/playlist?list=PL123",
        "not a url",
    ],
)
def test_can_compile_rejects_non_youtube(gemini_enabled, url):
    compiler = GeminiVideoCompiler(make_ctx(url))
    assert compiler.can_compile() is False


@pytest.mark.asyncio
@patch("mealie.services.recipe.import_workflow.compilers.gemini_video.GeminiService")
async def test_compile_returns_document_with_thumbnail(mock_service_cls, gemini_enabled):
    mock_service_cls.return_value.transcribe_youtube_video = AsyncMock(return_value="# Title\n\ntranscript")

    compiler = GeminiVideoCompiler(make_ctx("https://www.youtube.com/watch?v=abc123DEF45"))
    compiled = await compiler.compile()

    assert compiled is not None
    assert compiled.contains_recipe is True
    assert compiled.content == "# Title\n\ntranscript"
    assert compiled.image_url == "https://i.ytimg.com/vi/abc123DEF45/hqdefault.jpg"


@pytest.mark.asyncio
@patch("mealie.services.recipe.import_workflow.compilers.gemini_video.GeminiService")
async def test_compile_returns_none_on_error(mock_service_cls, gemini_enabled):
    mock_service_cls.return_value.transcribe_youtube_video = AsyncMock(side_effect=RuntimeError("boom"))

    compiler = GeminiVideoCompiler(make_ctx("https://youtu.be/abc123DEF45"))
    assert await compiler.compile() is None


@pytest.mark.asyncio
@patch("mealie.services.recipe.import_workflow.compilers.gemini_video.GeminiService")
async def test_compile_returns_none_on_empty_transcript(mock_service_cls, gemini_enabled):
    mock_service_cls.return_value.transcribe_youtube_video = AsyncMock(return_value=None)

    compiler = GeminiVideoCompiler(make_ctx("https://youtu.be/abc123DEF45"))
    assert await compiler.compile() is None
