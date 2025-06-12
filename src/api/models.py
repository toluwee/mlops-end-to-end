from pydantic import BaseModel, ConfigDict
from typing import List, Optional

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }
    )
    features: List[float]

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 0,
                "probability": [0.9, 0.05, 0.05],
                "request_id": "req_123456789"
            }
        }
    )
    prediction: int
    probability: Optional[List[float]] = None
    request_id: str
