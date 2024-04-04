"""Function(s) for cleaning the data set(s)."""


def clean_data(data, data_info):
    """Clean the data set(s).

    Args:
        data (pd.DataFrame): The data set(s) to clean.

    Returns:
        pd.DataFrame: The cleaned data set(s).

    """
    return _create_interaction_terms(data, data_info)


def _create_interaction_terms(data, data_info):
    """Create interaction terms for the data set(s).

    Args:
        data (pd.DataFrame): The data set(s) to create interaction terms for.

    Returns:
        pd.DataFrame: The data set(s) with the interaction terms.

    """
    data_select_1 = data[data_info["categorical_columns"][1]]
    data_select_2 = data[data_info["categorical_columns"][2]]
    data.loc[:, "interaction"] = data_select_1 * data_select_2
    return data
