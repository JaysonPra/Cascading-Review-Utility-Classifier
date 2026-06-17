from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data"
TOKEN_PATH = DATA_DIR / "continuation_token.pkl"
