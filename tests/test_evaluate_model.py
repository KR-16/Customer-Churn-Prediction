from src.data_loader.load_data import load_data
from src.preprocessing.handle_missing import clean_data
from src.preprocessing.encode_features import map_target
from src.utils.data_split import split_data
from src.pipeline import get_pipeline
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model

def test_evaluate_model_returns_metrics():
    df = map_target(clean_data(load_data()))
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = get_pipeline("Logistic_Regression")
    trained_pipeline = train_model(pipeline, X_train, y_train)
    metrics = evaluate_model(trained_pipeline, X_test, y_test)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "classification_report" in metrics