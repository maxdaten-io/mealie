import re
from typing import NamedTuple

import httpx

from mealie.core.config import get_app_settings
from mealie.core.root_logger import get_logger
from mealie.schema.openai.compiled_source import OpenAICompiledSource
from mealie.services.gemini import GeminiService

from .base import SourceCompiler, SourceType

YOUTUBE_URL_PATTERN = re.compile(
    r"^https?://(?:(?:www\.|m\.|music\.)?youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)

YOUTUBE_VIDEOS_API = "https://www.googleapis.com/youtube/v3/videos"


def youtube_video_id(url: str) -> str | None:
    match = YOUTUBE_URL_PATTERN.match(url)
    return match.group(1) if match else None


class VideoMetadata(NamedTuple):
    title: str
    description: str


async def fetch_video_description(video_id: str, api_key: str) -> VideoMetadata | None:
    """Fetches a video's title and description via the YouTube Data API. Returns None on
    any failure - the description tier is an optimization and must never block an import."""

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                YOUTUBE_VIDEOS_API, params={"part": "snippet", "id": video_id, "key": api_key}
            )
            response.raise_for_status()
            items = response.json().get("items") or []
    except Exception:
        get_logger().exception("YouTube Data API lookup failed for %s", video_id)
        return None

    if not items:
        return None

    snippet = items[0].get("snippet") or {}
    return VideoMetadata(title=snippet.get("title") or "", description=snippet.get("description") or "")


class GeminiVideoCompiler(SourceCompiler):
    """
    Compiles a YouTube video into a source document via Gemini, cheapest tier first:

    1. If the video's description (YouTube Data API) already holds a complete recipe -
       judged by a small text-only Gemini call - it becomes the document and the video
       is never watched (creators often paste the full recipe there).
    2. Otherwise Gemini watches the video natively (the URL is passed as file_data;
       Google fetches it server-side, which sidesteps yt-dlp's datacenter-IP 403s).
       A fetched description is merged in front of the transcript and marked
       authoritative, since pasted quantities beat transcribed speech.

    Runs before TranscriptionCompiler; only YouTube URLs are claimed, and any failure
    returns None so the workflow falls through to the yt-dlp path.
    """

    source_type = SourceType.URL
    # deliberately reuses an existing translated key - no frontend locale changes needed
    progress_key = "recipe.create-progress.transcribing-audio-with-ai"

    def can_compile(self) -> bool:
        url = self.ctx.input.url
        if not url:
            return False

        if not get_app_settings().GEMINI_READY:
            return False

        return youtube_video_id(url) is not None

    async def compile(self) -> OpenAICompiledSource | None:
        url = self.ctx.input.url or ""
        settings = get_app_settings()
        service = GeminiService(api_key=settings.GEMINI_API_KEY or "", model=settings.GEMINI_MODEL)
        video_id = youtube_video_id(url)

        meta = None
        if settings.YOUTUBE_API_KEY and video_id:
            meta = await fetch_video_description(video_id, settings.YOUTUBE_API_KEY)

        description = meta.description.strip() if meta else ""
        if description and meta:
            if compiled := await self._compile_from_description(service, meta, description, video_id):
                return compiled

        return await self._compile_from_video(service, url, meta, description, video_id)

    async def _compile_from_description(
        self, service: GeminiService, meta: VideoMetadata, description: str, video_id: str | None
    ) -> OpenAICompiledSource | None:
        """Tier 1: the description alone, if Gemini judges it complete enough to cook from."""

        try:
            assessment = await service.assess_description(description)
        except Exception:
            self.logger.exception("Gemini description assessment failed, treating description as incomplete")
            return None

        if not (assessment and assessment.has_ingredients and assessment.has_instructions):
            return None

        self.logger.info("YouTube description contains the full recipe, skipping video ingestion")
        content = f"# {meta.title}\n\n## Video description\n\n{description}"
        return self._compiled_source(content, video_id)

    async def _compile_from_video(
        self,
        service: GeminiService,
        url: str,
        meta: VideoMetadata | None,
        description: str,
        video_id: str | None,
    ) -> OpenAICompiledSource | None:
        """Tier 2: Gemini watches the video; a fetched description rides along as authoritative."""

        try:
            transcript = await service.transcribe_youtube_video(url)
        except Exception:
            self.logger.exception("Gemini video transcription failed, falling back to the next compiler")
            return None

        if not transcript:
            self.logger.error("Gemini returned no transcript for %s", url)
            return None

        if description and meta:
            content = "\n\n".join(
                [
                    "The official video description below is authoritative where it disagrees with the transcript.",
                    f"# {meta.title}",
                    f"## Video description\n\n{description}",
                    transcript,
                ]
            )
        else:
            content = transcript

        return self._compiled_source(content, video_id)

    def _compiled_source(self, content: str, video_id: str | None) -> OpenAICompiledSource:
        return OpenAICompiledSource(
            contains_recipe=True,
            content=content,
            language=None,
            # hqdefault exists for every video, unlike the higher-resolution variants
            image_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None,
        )
