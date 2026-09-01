import pandas as pd

from classifier_core.core.crud import get_reviews_with_llm_labels
from classifier_core.core.db import get_session
from classifier_core.core.types import ReviewLabelType


def convert_label_bouncer(df: pd.DataFrame) -> pd.DataFrame:
    conversion_dict: dict[ReviewLabelType, int] = {
        ReviewLabelType.SPAM: 1,
        ReviewLabelType.HIGH_UTILITY: 0,
        ReviewLabelType.LOW_UTILITY: 0,
    }

    df["is_spam"] = df["label"].map(conversion_dict)

    return df


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame: ...
def fit_tfidf(df: pd.DataFrame) -> None: ...


if __name__ == "__main__":
    with get_session() as session:
        reviews = get_reviews_with_llm_labels(session)

        reviews_data = [review.model_dump() for review in reviews]
        reviews_df = pd.DataFrame(reviews_data)

        converted = convert_label_bouncer(reviews_df)
