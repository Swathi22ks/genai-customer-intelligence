"""
preprocessor.py
===============
Data loading and preprocessing for all pipeline modules.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")


class CustomerDataPreprocessor:
    """Preprocessing pipeline for customer feature store."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    def load(self, path=None):
        path = path or f"{DATA_DIR}/feature_store.csv"
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df):,} customer records from {path}")
        return df

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()

        # Drop non-feature columns
        drop_cols = ["customer_id", "name", "email", "signup_date"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # Encode categoricals
        cat_cols = ["city", "state", "preferred_category",
                    "payment_method", "tenure_segment"]
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Handle nulls
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Features and target
        target = "churned"
        feature_cols = [c for c in df.columns if c != target]
        self.feature_columns = feature_cols

        X = df[feature_cols].values
        y = df[target].values

        return X, y, feature_cols

    def split_and_scale(self, X, y, test_size=0.2, val_size=0.1):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size / (1 - test_size),
            random_state=42, stratify=y_train
        )

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        logger.info(
            f"Split: Train={len(y_train):,} | Val={len(y_val):,} | Test={len(y_test):,}"
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class ReviewDataPreprocessor:
    """Preprocessing for review text data."""

    def load(self, path=None):
        path = path or f"{DATA_DIR}/reviews.csv"
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df):,} reviews from {path}")
        return df

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not isinstance(text, str):
            return ""
        text = text.strip().lower()
        # Keep punctuation for transformer models
        return text

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()
        df["clean_text"] = df["review_text"].apply(self.clean_text)
        df["text_length"] = df["clean_text"].str.len()
        df["word_count"] = df["clean_text"].str.split().str.len()

        # Sentiment numeric mapping
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        df["sentiment_numeric"] = df["sentiment_label"].map(sentiment_map)

        return df
