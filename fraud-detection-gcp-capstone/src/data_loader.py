"""
Data loading script for the Fraud Detection GCP Capstone project.

This module loads the Credit Card Fraud Detection dataset.
"""

import pandas as pd


def load_credit_card_data(file_path: str) -> pd.DataFrame:
    """
    Load the Credit Card Fraud Detection dataset.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        pandas DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Return basic dataset summary information.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing dataset shape, column list, and missing value count.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": int(df.isnull().sum().sum())
    }


if __name__ == "__main__":
    print("Data loader module is ready.")
