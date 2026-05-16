"""
Preprocessing pipeline for the Fraud Detection GCP Capstone project.

This module prepares the Credit Card Fraud Detection dataset for modeling.
The scaler is fitted only on the training data to prevent data leakage.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df, target_column="Class", test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets using stratification.

    Args:
        df: Input DataFrame.
        target_column: Target column name. For this dataset, the target is 'Class'.
        test_size: Proportion of data to use for testing.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_features(X_train, X_test):
    """
    Scale features while preventing data leakage.

    The scaler is fitted only on X_train. X_test is transformed using the
    fitted training scaler.

    Args:
        X_train: Training features.
        X_test: Test features.

    Returns:
        X_train_scaled, X_test_scaled, fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
