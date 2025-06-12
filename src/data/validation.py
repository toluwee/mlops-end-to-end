import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation class"""

    def __init__(self, expected_features: List[str], feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """Initialize validator with expected features and ranges"""
        self.expected_features = expected_features
        self.feature_ranges = feature_ranges or {}

    def validate_input(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data"""
        errors = []
        warnings = []

        # Check for missing features
        missing_features = set(self.expected_features) - set(data.columns)
        if missing_features:
            errors.append(f"Missing expected features: {', '.join(missing_features)}")
            return {"is_valid": False, "errors": errors, "warnings": warnings}

        # Check for unexpected features
        unexpected_features = set(data.columns) - set(self.expected_features)
        if unexpected_features:
            warnings.append(f"Unexpected features found: {', '.join(unexpected_features)}")

        # Validate feature ranges
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature not in data.columns:
                continue

            values = data[feature]
            out_of_range = values[(values < min_val) | (values > max_val)]
            if not out_of_range.empty:
                errors.append(
                    f"Feature '{feature}' has {len(out_of_range)} values out of expected range "
                    f"[{min_val}, {max_val}]"
                )

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
