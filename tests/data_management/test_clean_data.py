from difference_in_difference_with_deep_learning.data_management import clean_data


def test_clean_data_with_empty_data():
    data = "data"
    data_info = "data_info"
    assert clean_data(data, data_info) == "data", "data_info"
