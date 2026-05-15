"""
Data loading script for the Fraud Detection GCP Capstone project.

This file will contain functions to load the Credit Card Fraud Detection dataset
from a local file path or mounted Google Colab location.
"""

import pandas as pd


def load_credit_card_data(file_path: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        pandas DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)


if __name__ == "__main__":
    print("Data loader module created successfully.")
