import logging
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.preprocessing.encode_features import build_preprocessor
from src.config.feature_config import (
    MODEL_NAMES, RANDOM_STATE, MAXIMUM_ITERATIONS, 
    N_ESTIMATORS, CLASS_WEIGHT, USE_LABEL_ENCODER,
    EVAL_METRIC, SCALE_POS_WEIGHT, PROBABILITY
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_pipeline(model_name: str = "Logistic_Regression") -> Pipeline:
    """
    Returns a complete pipeline with preprocessing and selected classifier

    Args:
        model_name (str): Model identifier from config.MODEL_NAMES
    
    Returns:
        Pipeline: Scikit-learn pipeline with preprocessing and model
    """
    try:
        logger.info("Building the Pipeline for Multiple Models..........")
        preprocessor = build_preprocessor()

        models: dict[str, Any] = {
            "Logistic_Regression": LogisticRegression(max_iter=MAXIMUM_ITERATIONS, class_weight=CLASS_WEIGHT),
            "Random_Forest": RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, class_weight= CLASS_WEIGHT),
            "XG_Boost": XGBClassifier(use_label_encoder=USE_LABEL_ENCODER, eval_metric = EVAL_METRIC, scale_pos_weight = SCALE_POS_WEIGHT),
            "SVM": SVC(probability=PROBABILITY),
            "KNN": KNeighborsClassifier()
        }

        if model_name not in MODEL_NAMES:
            logger.error(f"Invalid Model Name: {model_name}. Choose from: {MODEL_NAMES}")
            raise ValueError(f"Invalid Model name: {model_name}. Choose from: {MODEL_NAMES}")
        
        logger.info(f"Selected Model: %s: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", models[model_name])
        ])
        logger.info(f"Pipeline Built Successfully for model: {model_name}")
        return pipeline
    
    except Exception as e:
        logger.error(f"Error building pipeline: {e}")
        raise