import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling the missing values and type conversions

    Operations:
    1. Drop CustomerID column
    2. Replace whitespace in TotalCharges with NaN
    3. Convert TotalCharges to float
    4. Fill missing TotalCharges with mean

    Args:
        df (pd.DataFrame): Raw DataFrame
    
    Returns:
        pd.DataFrame: cleaned DataFrame
    """
    try:

        df = df.copy()
        # Drop Customer ID
        if "customerID" in df.columns:
            df.drop("customerID", axis = 1, inplace=True)
            logger.info("Dropped Column: customerID")
        
        # replace the white paces
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
            logger.info("Replaced the white spaces in TotalCharges column with NA")
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")
            logger.info("Converted TotalCharges column to float dtype")
        
        # filling the NaN values with mean
        if df['TotalCharges'].isnull().sum() > 0:
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
            logger.info("Filled missing values in TotalCharges with mean values")
        
        return df
    except Exception as e:
        logger.error(f"Error during cleaning the dataset {e}")
        raise
    