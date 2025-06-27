import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data():
    file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    try:
        logger.info("Attempting to load the Dataset from Kaggle Hub...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "blastchar/telco-customer-churn",
            file_path
        )
        logger.info("Dataset Successfully Loaded with shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Failed to Load the dataset s: %s", str(e))
        raise

# if __name__ == "__main__":
#     try:
#         df = load_data()
#         print("Data Loaded Successfully")
#         print(df.head())
#     except Exception as e:
#         logger.critical("Script Terminated due to an error.")