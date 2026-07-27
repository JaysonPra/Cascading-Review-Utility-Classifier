from pathlib import Path

import pandas as pd

from classifier_core.core.constants import DATA_DIR
from classifier_core.core.crud import save_batch_review_manual_label
from classifier_core.core.db import get_session
from classifier_core.core.types import ReviewLabelType


def read_labeled_csv(file_path: Path) -> dict[int, ReviewLabelType]:
    """Reads CSV and scans for manually labeled reviews only"""
    df = pd.read_csv(file_path)
    valid_labels = [e.value for e in ReviewLabelType]
    df_labeled = df[df["manual_label"].isin(valid_labels)]

    return {
        int(review_id): ReviewLabelType(label)
        for review_id, label in zip(df_labeled["id"], df_labeled["manual_label"])
    }


if __name__ == "__main__":
    file_path = DATA_DIR / "labeled_reviews.csv"
    reviews = read_labeled_csv(file_path)

    with get_session() as session:
        save_batch_review_manual_label(session, reviews)
