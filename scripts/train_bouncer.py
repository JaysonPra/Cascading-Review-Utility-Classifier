# pyright: reportGeneralTypeIssues=false
# type: ignore

import argparse
from pathlib import Path
from typing import Any

import mlflow
import optuna
import pandas as pd
import yaml
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from classifier_core.core.constants import CONFIG_DIR
from classifier_core.core.crud import get_reviews_with_only_llm_labels
from classifier_core.core.db import get_session
from classifier_core.models.bouncer import (
    convert_label_bouncer,
    extract_metadata,
    objective,
    train_xgboost,
)

METADATA_COLS = [
    "char_count",
    "uppercase_count",
    "uppercase_ratio",
    "word_count",
    "exclamation_count",
    "question_count",
    "avg_word_len",
]


def load_config(file_path: Path) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "xgboost_config.yaml",
        help="Path to configuration yaml file",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Override max optuna trials",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=None,
        help="Override max latency constraint in ms",
    )
    return parser.parse_args()


def fetch_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    with get_session() as session:
        reviews = get_reviews_with_only_llm_labels(session)

    df = pd.DataFrame([r.model_dump() for r in reviews]).drop(
        columns=["manual_label"], errors="ignore"
    )
    df = convert_label_bouncer(df).drop(columns=["label"])
    df = extract_metadata(df)

    y = df["is_spam"]
    X = df.drop(columns=["is_spam"])
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, random_state=42, test_size=0.20, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.20,
        random_state=42,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[Any, Any, Any, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")

    X_train_text = vectorizer.fit_transform(X_train["content"])
    X_val_text = vectorizer.transform(X_val["content"])
    X_test_text = vectorizer.transform(X_test["content"])

    X_train_full = hstack([X_train_text, X_train[METADATA_COLS].values]).tocsr()
    X_val_full = hstack([X_val_text, X_val[METADATA_COLS].values]).tocsr()
    X_test_full = hstack([X_test_text, X_test[METADATA_COLS].values]).tocsr()

    return X_train_full, X_val_full, X_test_full, vectorizer


def start_experiment(
    X_train: Any,
    y_train: pd.Series,
    X_val: Any,
    y_val: pd.Series,
    X_test: Any,
    y_test: pd.Series,
    user_params: dict[str, list[int] | list[float]],
    vectorizer: TfidfVectorizer,
    max_latency_in_ms: float = 50.0,
    max_trials: int = 20,
) -> None:
    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val, user_params, max_latency_in_ms
        ),
        n_trials=max_trials,
    )

    best_classifier = train_xgboost(study.best_params, X_train, y_train, X_val, y_val)

    val_preds = best_classifier.predict(X_test)
    final_precision = float(precision_score(y_test, val_preds, zero_division=0))

    mlflow.log_params(study.best_params)
    mlflow.log_metric("val_precision", final_precision)

    mlflow.xgboost.log_model(best_classifier, artifact_path="bouncer_model")
    mlflow.sklearn.log_model(vectorizer, artifact_path="bouncer_vectorizer")


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_config(args.config)

    max_trials = (
        args.max_trials
        if args.max_trials is not None
        else config_dict.get("max_trials", 20)
    )
    max_latency = (
        args.max_latency
        if args.max_latency is not None
        else config_dict.get("max_latency_in_ms", 50.0)
    )

    X, y = fetch_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train_full, X_val_full, X_test_full, vectorizer = vectorize_features(
        X_train, X_val, X_test
    )

    start_experiment(
        X_train=X_train_full,
        y_train=y_train,
        X_val=X_val_full,
        y_val=y_val,
        X_test=X_test_full,
        y_test=y_test,
        user_params=config_dict["user_params"],
        vectorizer=vectorizer,
        max_latency_in_ms=max_latency,
        max_trials=max_trials,
    )
