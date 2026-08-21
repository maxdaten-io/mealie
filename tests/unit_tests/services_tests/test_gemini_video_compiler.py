from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mealie.core.config import get_app_settings
from mealie.services.gemini import DescriptionAssessment
from mealie.services.recipe.import_workflow.compilers import (
    DEFAULT_SOURCE_COMPILERS,
    GeminiVideoCompiler,
    TranscriptionCompiler,
)
from mealie.services.recipe.import_workflow.compilers.gemini_video import VideoMetadata, fetch_video_description
from mealie.services.recipe.import_workflow.context import WorkflowContext, WorkflowInput, WorkflowOptions

COMPILER_MODULE = "mealie.services.recipe.import_workflow.compilers.gemini_video"


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


# --- description-first tiering ---

URL = "https://www.youtube.com/watch?v=abc123DEF45"


@pytest.fixture()
def youtube_api_key(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-key")
    get_app_settings.cache_clear()


def make_service(transcript="## transcript body", assessment=None, assess_error=None):
    service = MagicMock()
    service.transcribe_youtube_video = AsyncMock(return_value=transcript)
    if assess_error:
        service.assess_description = AsyncMock(side_effect=assess_error)
    else:
        service.assess_description = AsyncMock(return_value=assessment)
    return service


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.fetch_video_description", new_callable=AsyncMock)
@patch(f"{COMPILER_MODULE}.GeminiService")
async def test_complete_description_skips_video(mock_service_cls, mock_fetch, gemini_enabled, youtube_api_key):
    mock_fetch.return_value = VideoMetadata(title="Pasta", description="Ingredients: 200g flour\nSteps: mix")
    service = make_service(assessment=DescriptionAssessment(has_ingredients=True, has_instructions=True))
    mock_service_cls.return_value = service

    compiled = await GeminiVideoCompiler(make_ctx(URL)).compile()

    assert compiled is not None
    assert "Pasta" in compiled.content
    assert "Ingredients: 200g flour" in compiled.content
    assert compiled.image_url == "https://i.ytimg.com/vi/abc123DEF45/hqdefault.jpg"
    service.transcribe_youtube_video.assert_not_awaited()


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.fetch_video_description", new_callable=AsyncMock)
@patch(f"{COMPILER_MODULE}.GeminiService")
async def test_partial_description_watches_video_and_merges(
    mock_service_cls, mock_fetch, gemini_enabled, youtube_api_key
):
    mock_fetch.return_value = VideoMetadata(title="Pasta", description="Ingredients: 200g flour")
    service = make_service(assessment=DescriptionAssessment(has_ingredients=True, has_instructions=False))
    mock_service_cls.return_value = service

    compiled = await GeminiVideoCompiler(make_ctx(URL)).compile()

    assert compiled is not None
    service.transcribe_youtube_video.assert_awaited_once()
    assert "Ingredients: 200g flour" in compiled.content
    assert "## transcript body" in compiled.content
    assert compiled.content.index("Ingredients: 200g flour") < compiled.content.index("## transcript body")


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.fetch_video_description", new_callable=AsyncMock)
@patch(f"{COMPILER_MODULE}.GeminiService")
async def test_no_api_key_goes_straight_to_video(mock_service_cls, mock_fetch, gemini_enabled):
    service = make_service()
    mock_service_cls.return_value = service

    compiled = await GeminiVideoCompiler(make_ctx(URL)).compile()

    assert compiled is not None
    mock_fetch.assert_not_awaited()
    service.assess_description.assert_not_awaited()
    assert compiled.content == "## transcript body"


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.fetch_video_description", new_callable=AsyncMock)
@patch(f"{COMPILER_MODULE}.GeminiService")
async def test_fetch_failure_falls_back_to_video(mock_service_cls, mock_fetch, gemini_enabled, youtube_api_key):
    mock_fetch.return_value = None
    service = make_service()
    mock_service_cls.return_value = service

    compiled = await GeminiVideoCompiler(make_ctx(URL)).compile()

    assert compiled is not None
    service.assess_description.assert_not_awaited()
    assert compiled.content == "## transcript body"


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.fetch_video_description", new_callable=AsyncMock)
@patch(f"{COMPILER_MODULE}.GeminiService")
async def test_assess_failure_still_merges_description(mock_service_cls, mock_fetch, gemini_enabled, youtube_api_key):
    mock_fetch.return_value = VideoMetadata(title="Pasta", description="Ingredients: 200g flour")
    service = make_service(assess_error=RuntimeError("quota"))
    mock_service_cls.return_value = service

    compiled = await GeminiVideoCompiler(make_ctx(URL)).compile()

    assert compiled is not None
    service.transcribe_youtube_video.assert_awaited_once()
    assert "Ingredients: 200g flour" in compiled.content
    assert "## transcript body" in compiled.content


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.fetch_video_description", new_callable=AsyncMock)
@patch(f"{COMPILER_MODULE}.GeminiService")
async def test_empty_description_skips_assessment(mock_service_cls, mock_fetch, gemini_enabled, youtube_api_key):
    mock_fetch.return_value = VideoMetadata(title="Pasta", description="  ")
    service = make_service()
    mock_service_cls.return_value = service

    compiled = await GeminiVideoCompiler(make_ctx(URL)).compile()

    assert compiled is not None
    service.assess_description.assert_not_awaited()
    assert compiled.content == "## transcript body"


# --- fetch_video_description ---


def make_httpx_client(response=None, error=None):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    if error:
        client.get = AsyncMock(side_effect=error)
    else:
        client.get = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.httpx.AsyncClient")
async def test_fetch_video_description_returns_metadata(mock_client_cls):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"items": [{"snippet": {"title": "Pasta", "description": "flour and water"}}]}
    client = make_httpx_client(response=response)
    mock_client_cls.return_value = client

    meta = await fetch_video_description("abc123DEF45", "yt-key")

    assert meta == VideoMetadata(title="Pasta", description="flour and water")
    kwargs = client.get.call_args.kwargs
    # the key must travel as a header, not in the URL - query params end up in access logs
    assert kwargs["headers"] == {"X-goog-api-key": "yt-key"}
    assert "key" not in kwargs["params"]


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.httpx.AsyncClient")
async def test_fetch_video_description_none_when_video_not_found(mock_client_cls):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"items": []}
    mock_client_cls.return_value = make_httpx_client(response=response)

    assert await fetch_video_description("abc123DEF45", "yt-key") is None


@pytest.mark.asyncio
@patch(f"{COMPILER_MODULE}.httpx.AsyncClient")
async def test_fetch_video_description_none_on_http_error(mock_client_cls):
    mock_client_cls.return_value = make_httpx_client(error=RuntimeError("network down"))

    assert await fetch_video_description("abc123DEF45", "yt-key") is None
