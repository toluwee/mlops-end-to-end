import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit preprocessor and transform data"""
        self.feature_names = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels if provided
        y_encoded = None
        if y is not None:
            y_encoded = self.label_encoder.fit_transform(y)

        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {X.shape[0]} samples with {X.shape[1]} features")

        return X_scaled, y_encoded

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if X.columns.tolist() != self.feature_names:
            raise ValueError(f"Feature mismatch. Expected: {self.feature_names}, Got: {X.columns.tolist()}")

        return self.scaler.transform(X)
