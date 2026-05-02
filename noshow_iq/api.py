from fastapi import FastAPI, Request
from pydantic import BaseModel 
from noshow_iq.database import predictions_col
import datetime
from typing import List, Optional

app = FastAPI()

class AppointmentInput(BaseModel):
    PatientId: str
    AppointmentID: str
    Gender: str
    ScheduledDay: str
    AppointmentDay: str
    Age: int
    Neighbourhood: str
    Scholarship: int
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    SMS_received: int

@app.get("/")
async def root():
    """Root endpoint - returns API status and available endpoints"""
    return {
        "message": "NoShowIQ API is running",
        "status": "healthy",
        "version": "0.1.0",
        "description": "Medical appointment no-show prediction API",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "history": "GET /history",
            "stats": "GET /stats"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "message": "API is healthy"}

@app.post("/predict")
async def predict(request: Request):
    # Receive the JSON data (text/audio paths)
    data = await request.json()
    
    # --- Model Logic Simulation ---
    # These would normally come from your model.py processing
    risk = "High" 
    prob = 0.6999
    advice = "Patient likely to miss appointment. Send SMS reminder."
    features = ["text_vec_0.12", "audio_freq_440"] # Simulated cleaned features
    
    # Q4 Requirement: Log every call to MongoDB Atlas
    prediction_entry = {
        "timestamp": datetime.datetime.utcnow(),
        "raw_input": data,
        "cleaned_features": features,
        "risk_level": risk,
        "probability": prob,
        "recommendation": advice
    }
    predictions_col.insert_one(prediction_entry)
    
    return {
        "risk_level": risk, 
        "probability": prob, 
        "recommendation": advice
    }

@app.get("/history")
async def get_history(limit: int = 20):
    """Get last N predictions from MongoDB"""
    try:
        history = list(
            predictions_col.find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        for record in history:
            if "timestamp" in record:
                record["timestamp"] = record["timestamp"].isoformat()
        
        return {
            "count": len(history),
            "predictions": history,
            "limit": limit
        }
    except Exception as e:
        return {
            "error": str(e),
            "count": 0,
            "predictions": []
        }

@app.get("/stats")
async def get_stats():
    # Q4 Requirement: MongoDB aggregation pipeline only, no Python computation
    pipeline = [
        {
            "$facet": {
                "total": [{"$count": "count"}],
                "by_risk": [{"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}],
                "avg_prob": [{"$group": {"_id": None, "avg": {"$avg": "$probability"}}}]
            }
        }
    ]
    
    # Execute aggregation
    agg_result = list(predictions_col.aggregate(pipeline))[0]
    
    # Format the output for the GET /stats requirement
    return {
        "total_predictions": agg_result["total"][0]["count"] if agg_result["total"] else 0,
        "high_risk_count": next((x["count"] for x in agg_result["by_risk"] if x["_id"] == "High"), 0),
        "low_risk_count": next((x["count"] for x in agg_result["by_risk"] if x["_id"] == "Low"), 0),
        "average_probability": round(agg_result["avg_prob"][0]["avg"], 2) if agg_result["avg_prob"] else 0.0,
        "last_trained": "2026-05-01T09:00:00Z" 
    }