import logging
from src.data_loader.load_data import load_data
from src.preprocessing.handle_missing import clean_data
from src.preprocessing.encode_features import map_target
from src.utils.data_split import split_data
from src.utils.results import save_metrics
from src.pipeline import get_pipeline
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.config.feature_config import MODEL_NAMES
from src.mlflow.run_experiments import run_all_experiments


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Churn.log"),
        logging.StreamHandler() # Also log console
    ]
)

logger = logging.getLogger(__name__)

def main():
    print("Hello from customer-churn-prediction!")
    logger.info("Starting the Customer Churn Prediction........")
    df = load_data()
    df_clean = clean_data(df)
    df_mapped = map_target(df_clean)
    X_train, X_test, y_train, y_test = split_data(df_mapped)

    for model_name in MODEL_NAMES:
        logger.info(f"Training and evaluating model: {model_name}")
        pipeline = get_pipeline(model_name)
        trained_pipeline = train_model(pipeline, X_train, y_train)
        metrics = evaluate_model(trained_pipeline, X_test, y_test)
        save_metrics(metrics, model_name)
    
    logger.info("Launching the MLFlow experiement")
    run_all_experiments()
    logger.info("Churn Prediction Completed Successfully")


if __name__ == "__main__":
    main()
