"""Tasks running the results formatting (tables, figures)."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from difference_in_difference_with_deep_learning.config import BLD
from difference_in_difference_with_deep_learning.final.did_simple_results import (
    generate_regression_table,
)


@pytask.mark.depends_on(
    {
        "did_regression_output": BLD
        / "difference_in_difference_with_deep_learning"
        / "models"
        / "did_model_output.pkl",
    },
)
@pytask.mark.task
@pytask.mark.produces(BLD / "python" / "tables" / "did_simple_reg_output.tex")
def task_create_results_table_simple_did(depends_on, produces):
    """Store a table in LaTeX format with the estimation results (Python version)."""
    model_results = pd.read_pickle(depends_on["did_regression_output"])
    model = generate_regression_table(model_results)
    with open(produces, "w") as fh:
        fh.write(model.as_latex())


def task_create_year_table_simple_did(
    did_table_year_output: Path = BLD
    / "difference_in_difference_with_deep_learning"
    / "models"
    / "did_table_year_output.pkl",
    table_year_output: Annotated[Path, Product] = BLD
    / "python"
    / "tables"
    / "did_table_year_output.tex",
) -> None:
    model_results = pd.read_pickle(did_table_year_output)
    with open(table_year_output, "w") as fh:
        fh.write(model_results.to_latex())
