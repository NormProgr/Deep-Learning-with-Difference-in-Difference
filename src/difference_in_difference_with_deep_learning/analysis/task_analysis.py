"""Tasks running the core analyses."""

import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
import yaml
from pytask import Product

from difference_in_difference_with_deep_learning.analysis import (
    did_regression,
    estimate_regression,
)
from difference_in_difference_with_deep_learning.config import BLD, SRC


def task_did_regression(
    cleaned_data: Path = BLD
    / "difference_in_difference_with_deep_learning"
    / "data"
    / "cleaned_data.csv",
    data_info: Path = SRC / "data_management" / "data_info.yaml",
    did_model_output: Annotated[Path, Product] = BLD
    / "difference_in_difference_with_deep_learning"
    / "models"
    / "did_model_output.pkl",
) -> None:
    data = pd.read_csv(cleaned_data)
    data_info = yaml.safe_load(open(data_info))
    regression_results = did_regression(data, data_info)

    with open(did_model_output, "wb") as f:
        pickle.dump(regression_results, f)


def task_estimate_regression(
    cleaned_data: Path = BLD
    / "difference_in_difference_with_deep_learning"
    / "data"
    / "cleaned_data.csv",
    data_info: Path = SRC / "data_management" / "data_info.yaml",
    did_table_year_output: Annotated[Path, Product] = BLD
    / "difference_in_difference_with_deep_learning"
    / "models"
    / "did_table_year_output.pkl",
) -> None:
    data = pd.read_csv(cleaned_data)
    data_info = yaml.safe_load(open(data_info))
    table_results = estimate_regression(data, data_info)
    with open(did_table_year_output, "wb") as f:
        pickle.dump(table_results, f)
