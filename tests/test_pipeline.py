from sklearn.pipeline import Pipeline
from src.pipeline import get_pipeline
from src.config.feature_config import MODEL_NAMES

def test_get_pipelines_returns_valid_pipelines():
    for model_name in MODEL_NAMES:
        pipeline = get_pipeline(model_name)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps