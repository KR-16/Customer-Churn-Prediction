import os
import mlflow
import logging
import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from src.config.feature_config import MODEL_NAMES, ARTIFACTS_DIR
from src.pipeline import get_pipeline
from src.data_loader.load_data import load_data
from src.preprocessing.handle_missing import clean_data
from src.preprocessing.encode_features import map_target
from src.utils.data_split import split_data
from src.utils.results import save_metrics
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def log_plots(y_test, y_pred, y_proba, model_name):
    """
    Logs confusion matrix, ROC Curve as MLFlow artifacts
    """
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    cm_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)


    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        roc_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)


def run_all_experiments():
    logger.info("Starting MLFlow experiments for all models.......")

    df = map_target(clean_data(load_data()))
    X_train, X_test, y_train, y_test = split_data(df)

    mlflow.set_experiment("Churn Experiment")
    best_model = None
    best_score = 0.0

    for model_name in MODEL_NAMES:
        logger.info(f"Running experiment for: {model_name}")
        pipeline = get_pipeline(model_name)

        with mlflow.start_run(run_name=model_name):
            # log hyperparameters
            mlflow.log_param("model", model_name)

            # Optional: add more specific model params here if needed

            # train and evaluate
            trained_pipeline = train_model(pipeline, X_train, y_train, model_name)
            metrics = evaluate_model(trained_pipeline, X_test, y_test, verbose=False)

            # Log metrics
            mlflow.log_metric("accuracy", metrics["accuracy"])
            if metrics["roc_auc"] is not None:
                mlflow.log_metric("roc_auc", metrics["roc_auc"])
            
            # save and log classification report
            report_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_report.json")
            save_metrics(metrics, filename = report_path)
            mlflow.log_artifact(report_path)

            # log confusion matrix and roc curve
            y_pred = trained_pipeline.predict(X_test)
            y_proba = trained_pipeline.predict_proba(X_test)[:,1] if hasattr(
                trained_pipeline.named_steps["classifier"], "predict_proba") else None
            log_plots(y_test=y_test, y_pred=y_pred, y_proba=y_proba, model_name=model_name)

            # log trained model
            mlflow.sklearn.log_model(trained_pipeline, artifact_path=model_name)

            # check for the best model
            if metrics["roc_auc"] is not None and metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = model_name
        logger.info(f"Completed run for : {model_name}")
    logger.info(f"All experiments done. Best Model: {best_model} with ROC AUC: {best_score:.4f}")