from google import genai
from google.genai import types

from classifier_core.core.config import settings
from classifier_core.schemas.label import ReviewBatchResponse


def create_prompt(review_batch: list[str], prompt: str) -> str:
    for idx, review in enumerate(review_batch):
        prompt += f"\n{idx}. {review}\n"

    return prompt


def label_review(client: genai.Client, prompt_content: str) -> str | None:
    response = client.models.generate_content(
        contents=prompt_content,
        model=settings.label_model,
        config=types.GenerateContentConfig(
            response_schema=ReviewBatchResponse,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    return response.text
