from src.data_loader.load_data import load_data
from src.preprocessing.handle_missing import clean_data
from src.preprocessing.encode_features import map_target
from src.utils.data_split import split_data
from src.pipeline import get_pipeline
from src.models.train_model import train_model

def test_train_model_fits_pipeline():
    df = map_target(clean_data(load_data(df)))

    X_train, _, y_train, _ = split_data(df)

    pipeline = get_pipeline("Logistic_Regression")
    trained_pipeline = train_model(pipeline, X_train, y_train)

    assert hasattr(trained_pipeline, "predict")