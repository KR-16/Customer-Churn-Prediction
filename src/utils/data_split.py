from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.feature_config import target_column, TEST_SIZE, RANDOM_STATE
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets

    Args:
        df (pd.DataFrame): The cleaned dataset including the target column
        test_size (float): Proportion of the dataset to include in the test size
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    try:
        logger.info("Splitting the Dataset.......")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        logger.info("Dataset split into X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error while splitting the data: {e}")
        raise