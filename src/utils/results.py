import os
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def save_metrics(metrics: dict, model_name: str = "", output_dir: str = "results", filename: str = None) -> None:
    """
    Save evaluation metrics to a JSON file under results directory

    Args:
        metrics (dict): Dictionary of evaluation metrics
        model_name (str): Model name to include in the filename
        output_sir (str): Directory to save the results (default: "results")
    """

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    try:
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
