"""
Preprocessing pipeline for the Fraud Detection GCP Capstone project.

This script will handle train/test splitting, scaling, and leakage prevention.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df, target_column="Class", test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.

    Args:
        df: Input DataFrame.
        target_column: Name of target column.
        test_size: Test split size.
        random_state: Random seed for reproducibility.

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
    Fit scaler on training data only to prevent data leakage.

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
