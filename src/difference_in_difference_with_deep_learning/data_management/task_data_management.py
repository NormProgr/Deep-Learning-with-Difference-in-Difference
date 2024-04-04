"""Tasks for managing the data."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from difference_in_difference_with_deep_learning.config import BLD, SRC
from difference_in_difference_with_deep_learning.data_management import clean_data


def task_clean_data(
    testdata: Path = SRC / "data" / "testdata.csv",
    data_info: Path = SRC / "data_management" / "data_info.yaml",
    cleaned_data: Annotated[Path, Product] = BLD
    / "difference_in_difference_with_deep_learning"
    / "data"
    / "cleaned_data.csv",
) -> None:
    data = pd.read_csv(testdata)
    data = clean_data(data)
    data.to_csv(cleaned_data, index=False)
