import sys

from loguru import logger

from classifier_core.core.constants import LOGS_DIR


def setup_console_logger() -> None:
    logger.add(sys.stderr, level="INFO")


# In your logger setup file:
def setup_ingestion_logger() -> None:
    logger.add(
        sink=LOGS_DIR / "ingestion" / "ingestion.log",
        level="DEBUG",
        serialize=True,
        filter=lambda record: (
            "ingest" in record["name"] or "ingest" in record["file"].name  # type: ignore
        ),
        rotation="00:00",
        retention="30 days",
        compression="zip",
        enqueue=True,
    )


def setup_training_logger() -> None:
    logger.add(
        sink=LOGS_DIR / "training" / "training.log",
        level="DEBUG",
        serialize=True,
        filter=lambda record: (
            "train" in record["name"] or "train" in record["file"].name  # type: ignore
        ),
        rotation="Monday at 00:00",
        retention="90 days",
        compression="zip",
        enqueue=True,
    )


def setup_logger() -> None:
    logger.remove()

    setup_console_logger()
    setup_ingestion_logger()
    setup_training_logger()
