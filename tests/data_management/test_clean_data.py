from difference_in_difference_with_deep_learning.data_management import clean_data


def test_clean_data():
    data = "data"
    assert clean_data(data) == data
