from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mealie.core.config import get_app_settings
from mealie.services.gemini import GeminiService


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_gemini_settings_default_disabled():
    settings = get_app_settings()
    assert settings.GEMINI_ENABLED is False
    assert settings.GEMINI_READY is False


def test_gemini_ready_requires_key(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "True")
    get_app_settings.cache_clear()
    assert get_app_settings().GEMINI_READY is False


def test_gemini_ready_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "True")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_app_settings.cache_clear()
    settings = get_app_settings()
    assert settings.GEMINI_READY is True
    assert settings.GEMINI_MODEL == "gemini-3.5-flash"


@pytest.mark.asyncio
@patch("mealie.services.gemini.gemini_service.genai.Client")
async def test_transcribe_youtube_video_passes_url_as_file_data(mock_client_cls):
    response = MagicMock()
    response.text = "# Title\n\n## Spoken recipe content\n..."
    generate = AsyncMock(return_value=response)
    mock_client_cls.return_value.aio.models.generate_content = generate

    service = GeminiService(api_key="test-key", model="test-model")
    result = await service.transcribe_youtube_video("https://www.youtube.com/watch?v=abc123DEF45")

    assert result == response.text
    kwargs = generate.call_args.kwargs
    assert kwargs["model"] == "test-model"
    file_parts = [p for p in kwargs["contents"].parts if p.file_data]
    assert len(file_parts) == 1
    assert file_parts[0].file_data.file_uri == "https://www.youtube.com/watch?v=abc123DEF45"


@pytest.mark.asyncio
@patch("mealie.services.gemini.gemini_service.genai.Client")
async def test_transcribe_youtube_video_returns_none_on_empty_response(mock_client_cls):
    response = MagicMock()
    response.text = None
    mock_client_cls.return_value.aio.models.generate_content = AsyncMock(return_value=response)

    service = GeminiService(api_key="test-key", model="test-model")
    assert await service.transcribe_youtube_video("https://youtu.be/abc123DEF45") is None
