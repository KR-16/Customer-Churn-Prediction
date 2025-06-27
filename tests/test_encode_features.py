import pandas as pd
from src.data_loader.load_data import load_data
from src.preprocessing.handle_missing import clean_data
from src.preprocessing.encode_features import map_target

def test_map_target_converts_churn_columns():
    df = load_data()
    cleaned_df = clean_data(df)
    mapped_df = map_target(cleaned_df)

    assert mapped_df["Churn"].dropna().isin([0,1]).all()