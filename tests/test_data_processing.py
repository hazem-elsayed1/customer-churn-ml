import pandas as pd

from src.data_processing import clean_data


def test_total_charges_is_converted_to_numeric():
    df = pd.DataFrame(
        {
            "tenure": [10],
            "TotalCharges": ["500.5"],
        }
    )

    cleaned_df = clean_data(df)

    assert cleaned_df["TotalCharges"].dtype.kind in "fi"
    assert cleaned_df.loc[0, "TotalCharges"] == 500.5


def test_zero_tenure_sets_total_charges_to_zero():
    df = pd.DataFrame(
        {
            "tenure": [0],
            "TotalCharges": [" "],
        }
    )

    cleaned_df = clean_data(df)

    assert cleaned_df.loc[0, "TotalCharges"] == 0