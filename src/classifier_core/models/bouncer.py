import pandas as pd

from classifier_core.core.types import ReviewLabelType


def convert_label_bouncer(df: pd.DataFrame) -> pd.DataFrame:
    conversion_dict: dict[ReviewLabelType, int] = {
        ReviewLabelType.SPAM: 1,
        ReviewLabelType.HIGH_UTILITY: 0,
        ReviewLabelType.LOW_UTILITY: 0,
    }

    df["is_spam"] = df["label"].map(conversion_dict)

    return df


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df["char_count"] = df["content"].str.len()

    df["uppercase_count"] = df["content"].str.findall(r"[A-Z]").str.len()
    df["uppercase_ratio"] = df["uppercase_count"] / df["char_count"].replace(0, 1)

    df["word_count"] = df["content"].str.split().str.len()
    df["exclamation_count"] = df["content"].str.count(r"!")
    df["question_count"] = df["content"].str.count(r"\?")

    char_count_no_spaces = df["content"].str.replace(r"\s+", "", regex=True).str.len()
    df["avg_word_len"] = char_count_no_spaces / df["word_count"].replace(0, 1)

    return df
