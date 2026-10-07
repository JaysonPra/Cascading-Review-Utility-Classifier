from pathlib import Path

import yaml

from classifier_core.core.constants import DATA_DIR

if __name__ == "__main__":
    file_path: Path = DATA_DIR / "params.yaml"

    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
