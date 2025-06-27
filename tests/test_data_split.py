import pandas as pd
from src.data_loader.load_data import load_data
from src.preprocessing.handle_missing import clean_data
from src.preprocessing.encode_features import map_target
from src.utils.data_split import split_data
from src.config.feature_config import target_column

def test_split_data_shape_types():
    df = map_target(clean_data(load_data()))
    X_train, X_test, y_train, y_test = split_data(df)
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert target_column not in X_train.columns