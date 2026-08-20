import re

from mealie.core.config import get_app_settings
from mealie.schema.openai.compiled_source import OpenAICompiledSource
from mealie.services.gemini import GeminiService

from .base import SourceCompiler, SourceType

YOUTUBE_URL_PATTERN = re.compile(
    r"^https?://(?:(?:www\.|m\.|music\.)?youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)


def youtube_video_id(url: str) -> str | None:
    match = YOUTUBE_URL_PATTERN.match(url)
    return match.group(1) if match else None


class GeminiVideoCompiler(SourceCompiler):
    """
    Compiles a YouTube video into a transcript document by having Gemini watch it natively
    (the URL is passed to the model as file_data; Google fetches the video server-side).

    Runs before TranscriptionCompiler because yt-dlp cannot download YouTube media from
    datacenter egress IPs. Only YouTube URLs are claimed - Gemini ingests nothing else -
    and any failure returns None so the workflow falls through to the yt-dlp path.
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

        try:
            transcript = await service.transcribe_youtube_video(url)
        except Exception:
            self.logger.exception("Gemini video transcription failed, falling back to the next compiler")
            return None

        if not transcript:
            self.logger.error("Gemini returned no transcript for %s", url)
            return None

        video_id = youtube_video_id(url)
        return OpenAICompiledSource(
            contains_recipe=True,
            content=transcript,
            language=None,
            # hqdefault exists for every video, unlike the higher-resolution variants
            image_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None,
        )
