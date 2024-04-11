import pandas as pd
import statsmodels.formula.api as smf


def did_regression(data, data_info):
    # Extract variables from data_info
    outcome_variable = data_info["outcome"]
    causal_effect_variable = data_info["causal_effect"]
    categorical_columns = data_info["categorical_columns"]
    control_columns = data_info["control_columns"]

    # Create formula string
    formula = f"{outcome_variable} ~ {causal_effect_variable} + {' + '.join(categorical_columns)}  + {' + '.join(control_columns)}"

    # Set up the regression model
    reg_model = smf.ols(formula=formula, data=data)

    # Fit the regression model
    results = reg_model.fit()

    # Return the summary of the regression results
    return results.summary2()


def estimate_regression(data, data_info):
    """Estimate regression models for each time period and summarize the results.

    Parameters:
        data (DataFrame): The dataset containing all variables.
        data_info (dict): Dictionary containing data configuration information.

    Returns:
        DataFrame: A DataFrame containing summary statistics for each time period.

    """
    results_list = []

    for time_period in data[data_info["time"]].unique():
        # Filter the data for the current time period
        data_time_period = data[data[data_info["time"]] == time_period]

        # Define the regression formula
        formula = f"{data_info['outcome']} ~ {data_info['causal_effect']} + {' + '.join(data_info['categorical_columns'] )} + {' + '.join(data_info['control_columns'])}"

        # Fit the regression model
        reg_model = smf.ols(formula=formula, data=data_time_period)
        results = reg_model.fit()

        # Extract coefficient, std. error, and p-value
        coefficient = results.params[data_info["causal_effect"]]
        std_error = results.bse[data_info["causal_effect"]]
        p_value = results.pvalues[data_info["causal_effect"]]

        # Calculate the control mean
        control_mean = data_time_period[data_info["outcome"]].mean()

        # Calculate the difference between treatment and control groups
        difference_tc = (
            data_time_period[data_time_period[data_info["causal_effect"]] == 1][
                data_info["outcome"]
            ].mean()
            - control_mean
        )

        # Calculate the difference with controls
        # Assume mean of interaction term for simplicity
        data_time_period[data_info["causal_effect"]].mean()
        difference_tc_controls = (
            difference_tc
            - coefficient
            * (
                data_time_period[data_info["control_columns"]]
                - data_time_period[data_info["control_columns"]].mean()
            )
            .mean()
            .sum()
        )

        # Append the results to the list
        results_list.append(
            {
                "Time Period": time_period,
                "Control Mean": control_mean,
                "Difference T-C": difference_tc,
                "Difference T-C with Controls": difference_tc_controls,
                "Coefficient": coefficient,
                "Std. Error": std_error,
                "P-value": p_value,
            },
        )

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(results_list)
