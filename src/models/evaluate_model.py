import logging
from typing import Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluate_model(pipeline: Pipeline, X_test, y_test, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluates a trained pipeline on test data

    Args:
        pipeline (Pipeline): Trained pipeline to evaluate
        X_test: Test feature set
        y_test: Test target values
        verbose (bool): Whether to print detailed output
    
        Returns:
            Dict[str, Any]: Evaluation metrics
    """
    try:
        logger.info("Evaluating the Model.........")
        y_pred = pipeline.predict(X_test)
        y_proba = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(pipeline.named_steps["classifier"], "predict_proba") else None
        )
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        if verbose:
            logger.info("Evaluation Result:")
            print(classification_report(y_test, y_pred))
            if y_proba is not None:
                print(f"ROC AUC Score: {metrics["roc_auc"]:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"Error while evaluating the model: {e}")
        raise