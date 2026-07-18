from pathlib import Path

import pandas as pd
from openpyxl.worksheet.datavalidation import DataValidation
from sqlmodel import Session

from classifier_core.core.constants import DATA_DIR
from classifier_core.core.crud import get_batch_reviews
from classifier_core.core.db import get_session


def create_excel_sheet(
    session: Session, file_path: Path, num_reviews: int = 150
) -> None:
    reviews = get_batch_reviews(session, num_reviews)

    df = pd.DataFrame([r.model_dump() for r in reviews])
    df = df[["id", "manual_label", "content"]]

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Reviews")  # type: ignore

        worksheet = writer.sheets["Reviews"]

        dv = DataValidation(
            type="list", formula1='"High Utility,Low Utility,Spam"', allow_blank=True
        )

        worksheet.add_data_validation(dv)

        dv.add("B2:B1000")  # type: ignore


if __name__ == "__main__":
    file_path = DATA_DIR / "reviews.xlsx"

    with get_session() as session:
        create_excel_sheet(session, file_path, num_reviews=500)
