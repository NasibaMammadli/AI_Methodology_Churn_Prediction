"""
FastAPI application for serving the RetailGenius churn prediction model.

This module provides:
- REST API endpoints for predictions
- Model health checks
- Batch prediction endpoints
- Model metadata endpoints
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import uvicorn

from .predict import ChurnPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RetailGenius Churn Prediction API",
    description="API for predicting customer churn at RetailGenius",
    version="1.0.0"
)

# Initialize predictor
predictor = None

# Pydantic models for request/response
class CustomerData(BaseModel):
    customer_id: str
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    paperless_billing: str
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    unlimited_data: str

class PredictionResponse(BaseModel):
    customer_id: str
    churn_prediction: int
    churn_probability: float
    confidence: float
    risk_level: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global predictor
    try:
        model_path = "models/best_model.pkl"
        if Path(model_path).exists():
            predictor = ChurnPredictor(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found. API will not be able to make predictions.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "RetailGenius Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData):
    """Make a prediction for a single customer."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        data = pd.DataFrame([customer.dict()])
        
        # Make prediction
        results = predictor.predict_with_confidence(data)
        
        # Return first (and only) result
        result = results.iloc[0]
        
        return PredictionResponse(
            customer_id=str(result['customer_id']),
            churn_prediction=int(result['churn_prediction']),
            churn_probability=float(result['churn_probability']),
            confidence=float(result['confidence']),
            risk_level=str(result['risk_level'])
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Make predictions for multiple customers from a CSV file."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        content = await file.read()
        data = pd.read_csv(pd.io.common.BytesIO(content))
        
        # Make predictions
        results = predictor.predict_with_confidence(data)
        
        # Convert to response format
        predictions = []
        for _, row in results.iterrows():
            predictions.append(PredictionResponse(
                customer_id=str(row['customer_id']),
                churn_prediction=int(row['churn_prediction']),
                churn_probability=float(row['churn_probability']),
                confidence=float(row['confidence']),
                risk_level=str(row['risk_level'])
            ))
        
        # Create summary
        summary = {
            "total_customers": len(results),
            "predicted_churn": int(results['churn_prediction'].sum()),
            "churn_rate": float(results['churn_prediction'].mean()),
            "average_probability": float(results['churn_probability'].mean()),
            "average_confidence": float(results['confidence'].mean()),
            "risk_distribution": results['risk_level'].value_counts().to_dict()
        }
        
        return BatchPredictionResponse(predictions=predictions, summary=summary)
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get feature importance
        importance_df = predictor.get_feature_importance()
        
        return {
            "model_path": predictor.model_path,
            "feature_count": len(predictor.feature_names),
            "feature_names": predictor.feature_names,
            "top_features": importance_df.head(10).to_dict('records')
        }
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/features")
async def get_feature_importance():
    """Get feature importance scores."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        importance_df = predictor.get_feature_importance()
        return {"feature_importance": importance_df.to_dict('records')}
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/custom")
async def predict_custom(data: List[Dict[str, Any]]):
    """Make predictions for custom data format."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        results = predictor.predict_with_confidence(df)
        
        return {
            "predictions": results.to_dict('records'),
            "summary": {
                "total_customers": len(results),
                "predicted_churn": int(results['churn_prediction'].sum()),
                "churn_rate": float(results['churn_prediction'].mean()),
                "average_probability": float(results['churn_probability'].mean())
            }
        }
    
    except Exception as e:
        logger.error(f"Error in custom prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 