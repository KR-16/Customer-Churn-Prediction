import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.config.feature_config import (
    binary_class_columns,
    multi_class_columns,
    numerical_columns
)
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the target column "Churn" from Yes/No to 1/0
    """

    try:
        logger.info("Mapping the target column!")
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes":0, "No": 1})
            logger.info("Mapped the Churn - Target Column to 1 and 0")
        return df
    except Exception as e:
        logger.error(f"Error during the Mapping: {e}")
        raise

def build_preprocessor() -> ColumnTransformer:
    """
    Build a Column transformer that:
    - imputes and ordinal-encodes binary categorical column
    - imputes and one-hot encodes multi-class categorical column
    - imputes and scales numerical columns

    Return:
        ColumnTransformer: the preprocessing pipeline
    """
    try:
        logger.info("Building the Preprocessor Pipeline........")
        binary_pipeline = Pipeline(
            steps = [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]
        )
        logger.info("Created a Binary Pipeline for Binary Columns")

        multi_class_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ]
        )
        logger.info("Created a Multi Class Pipeline for Multiple Columns")

        numerical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler())
            ]
        )
        logger.info("Created a Numerical Pipeline for Numerical Columns")

        preprocessor = ColumnTransformer(
            transformers=[
                ("binary", binary_pipeline, binary_class_columns),
                ("multi", multi_class_pipeline, multi_class_columns),
                ("numerical", numerical_pipeline, numerical_columns)
            ],
            remainder="drop"
        )
        logger.info("Built the Preprocessor using Column Transformer")
        return preprocessor
    
    except Exception as e:
        logger.error(f"Error occured while building preprocessor {e}")
        raise