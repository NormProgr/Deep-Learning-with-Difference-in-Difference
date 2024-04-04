"""Tasks for managing the data."""

import pandas as pd
import pytask

from difference_in_difference_with_deep_learning.config import BLD, SRC
from difference_in_difference_with_deep_learning.data_management import clean_data


@pytask.mark.depends_on(
    {
        "script": ["clean_data.py"],
        "data": SRC / "data" / "testdata.csv",
    },
)
@pytask.mark.task
@pytask.mark.produces(
    BLD / "difference_in_difference_with_deep_learning" / "data" / "cleaned_data.csv",
)
def task_clean_data(depends_on, produces):
    data = pd.read_csv(depends_on["data"])
    data = clean_data(data)
    data.to_csv(produces, index=False)
