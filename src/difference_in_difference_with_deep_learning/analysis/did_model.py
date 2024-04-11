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
