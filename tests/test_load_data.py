import pandas as pd
from src.data_loader.load_data import load_data

def test_load_data_returns_DataFrame():
    df = load_data()
    assert isinstance(df, pd.DataFrame)

def test_load_data_expected_columns():
    df = load_data()
    expected_columns = {"customerID", "Churn", "TotalCharges", "Dependents"}
    assert expected_columns.issubset(df.columns)