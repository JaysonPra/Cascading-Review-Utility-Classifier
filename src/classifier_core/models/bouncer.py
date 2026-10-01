import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import precision_score
from xgboost import XGBClassifier

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


def get_latencies(X_val: pd.DataFrame, classifier: XGBClassifier) -> list[float]:
    latencies: list[float] = []

    for i in range(1, min(len(X_val), 100)):
        sample = X_val.iloc[i]
        start = time.perf_counter()
        classifier.predict(sample)
        latencies.append((time.perf_counter() - start) * 1000)

    return latencies


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    user_params: dict[str, list[int] | list[float]],
) -> float:
    params: dict[str, int | float] = {
        "max_depth": trial.suggest_int("max_depth", *user_params["max_depth"]),  # type: ignore
        "min_child_weight": trial.suggest_float(
            "min_child_weight", *user_params["min_child_weight"]
        ),
        "subsample": trial.suggest_float("subsample", *user_params["subsample"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *user_params["learning_rate"]
        ),
        "n_estimators": trial.suggest_int("n_estimators", *user_params["n_estimators"]),  # type: ignore
    }

    classifier = XGBClassifier(**params, random_state=42)

    classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    latencies = get_latencies(X_val, classifier)
    p90_latency = np.percentile(latencies, 90)

    if p90_latency > params["max_latency_in_ms"]:
        raise optuna.TrialPruned()

    preds = classifier.predict(X_val)
    return float(precision_score(y_val, preds, zero_division=0))
