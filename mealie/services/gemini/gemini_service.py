from google import genai
from google.genai import types
from pydantic import BaseModel


class DescriptionAssessment(BaseModel):
    has_ingredients: bool
    has_instructions: bool


ASSESS_DESCRIPTION_PROMPT = """\
Below is the description of a cooking video. Judge whether it alone is enough to cook from:
- has_ingredients: it lists the recipe's ingredients, with quantities
- has_instructions: it contains preparation steps complete enough to follow

References to the video ("full recipe in the video"), links, or a bare dish name do not count.

Description:

"""

TRANSCRIBE_VIDEO_PROMPT = """\
Watch this cooking video and produce a faithful written record of it in Markdown:

# <video title>

## Video description
<the description or a one-paragraph summary of what the video is about>

## Spoken recipe content
<everything the narrator says that is relevant to the recipe: ingredients with \
quantities, steps, techniques, temperatures, timings - in order, as complete as possible>

Do not invent details that are not in the video. Answer in the language spoken in the video."""


class GeminiService:
    """Thin wrapper around the native google-genai SDK for multimodal features
    the OpenAI-compatible endpoint cannot provide (e.g. YouTube video ingestion)."""

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def transcribe_youtube_video(self, url: str) -> str | None:
        """Have Gemini watch a public YouTube video and return a Markdown transcript
        document, or None if the model returned no usable text."""

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=types.Content(
                role="user",
                parts=[
                    types.Part(file_data=types.FileData(file_uri=url, mime_type="video/*")),
                    types.Part.from_text(text=TRANSCRIBE_VIDEO_PROMPT),
                ],
            ),
            # low resolution: the recipe signal is in the audio, and video tokens
            # drop from ~300/s to ~100/s
            config=types.GenerateContentConfig(media_resolution="MEDIA_RESOLUTION_LOW"),
        )
        return response.text or None

    async def assess_description(self, description: str) -> DescriptionAssessment | None:
        """Cheap text-only call judging whether a video description already contains a
        complete recipe, so the caller can skip the far more expensive video ingestion."""

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=ASSESS_DESCRIPTION_PROMPT + description,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DescriptionAssessment,
            ),
        )
        return response.parsed
