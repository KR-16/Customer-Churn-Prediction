import logging
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """
    Trains the pipeline with the training data.

    Args:
        pipeline (Pipeline): Preprocessing + model pipeline
        X_train: Training feature set
        y_train: Training target values
    
    Returns:
        Pipeline: trained pipeline
    """
    try:
        logger.info("Training the Model........")
        pipeline.fit(X_train, y_train)
        logger.info("Model Training Complete.")
        return pipeline
    except Exception as e:
        logger.error(f"Error during Model Training {e}")
        raise