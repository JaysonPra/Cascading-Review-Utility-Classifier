from pathlib import Path

import yaml
from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel, Field

from classifier_core.core.config import settings
from classifier_core.schemas.database import Review
from classifier_core.schemas.label import ReviewBatchResponse


class LabelingJobConfig(BaseModel):
    batch_size: int = Field(default=20, ge=1)
    num_reviews: int = Field(default=100, ge=1)
    system_instruction: str

    @classmethod
    def load_from_yaml(cls, file_path: Path) -> "LabelingJobConfig":
        """Loads and validates the labeling job configuration from a YAML file."""
        logger.info(f"Loading configuration from {file_path}")
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            config = cls(**data)

            logger.success(
                f"Configuration loaded: batch_size={config.batch_size}, num_reviews={config.num_reviews}"
            )

            return config

        except Exception:
            logger.exception(f"Failed to load configuration from {file_path}")
            raise


def build_batch_prompt(review_batch: list[Review], system_instructions: str) -> str:
    """Formats a structured text prompt combining system instructions and a batch of reviews."""
    logger.info(f"Building batch prompt for {len(review_batch)} reviews")

    serialized_reviews = "\n".join(f"{r.id}. {r.content}" for r in review_batch)
    return f"{system_instructions}\nReviews to process:\n{serialized_reviews}"


def label_batch_reviews(client: genai.Client, prompt_content: str) -> str:
    """Sends the formatted prompt to the Gemini API and returns the validated structured JSON string."""
    logger.info(f"Sending batch request to Gemini model: {settings.label_model}")
    try:
        response = client.models.generate_content(
            contents=prompt_content,
            model=settings.label_model,
            config=types.GenerateContentConfig(
                response_schema=ReviewBatchResponse,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        if not response.text:
            logger.warning("Gemini returned an empty response text block.")
            return '{"batch_response": []}'

        logger.success("Successfully received batch response from Gemini")
        return response.text

    except Exception:
        logger.exception("API call to Gemini failed")
        return '{"batch_response": []}'
