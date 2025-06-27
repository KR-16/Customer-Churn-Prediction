import pandas as pd
from src.data_loader.load_data import load_data
from src.preprocessing.handle_missing import clean_data

def test_clean_data_removes_customer_id_nans():
    df = load_data()
    cleaned_df = clean_data(df)

    assert "customerID" not in cleaned_df.columns
    assert cleaned_df.isnull().sum().sum() == 0