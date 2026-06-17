import pickle

from google_play_scraper import Sort, reviews  # type: ignore
from google_play_scraper.features.reviews import _ContinuationToken  # type: ignore

from classifier_core.core.constants import DATA_DIR, TOKEN_PATH


def save_token(token: _ContinuationToken) -> None:
    if token:
        DATA_DIR.mkdir(exist_ok=True)

        with open(TOKEN_PATH, "wb") as file:
            pickle.dump(token, file)


def read_token() -> _ContinuationToken | None:
    try:
        with open(TOKEN_PATH, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


def start_ingestion() -> _ContinuationToken:
    continuation_token = read_token()

    review, continuation_token = reviews(  # type: ignore
        "com.valvesoftware.android.steam.community",
        sort=Sort.NEWEST,
        continuation_token=continuation_token,  # type: ignore
    )

    return continuation_token


if __name__ == "__main__":
    continuation_token = None

    try:
        while True:
            continuation_token = start_ingestion()

            if not continuation_token:
                print("No more reviews found...")
                break

    except KeyboardInterrupt:
        print("\nIngestion interrupted. Saving token...")
        if continuation_token:
            save_token(continuation_token)
