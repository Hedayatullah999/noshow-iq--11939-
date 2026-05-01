import os
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from noshow_iq.preprocess import clean_data, extract_features
from noshow_iq.model import predict

app = FastAPI(title="NoShowIQ API")

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    client = MongoClient(MONGO_URI)
    db = client.noshow_db
else:
    db = None

class Appointment(BaseModel):
    PatientId: float
    AppointmentID: int
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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def make_prediction(record: Appointment):
    # 1. Convert input to DataFrame
    df_raw = pd.DataFrame([record.model_dump()])
    
    # 2. Preprocess
    df_clean = clean_data(df_raw)
    features = extract_features(df_clean)
    
    # 3. Predict
    try:
        risk_level, probability = predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model not trained or not found.")

    recommendation = "Send SMS reminder and call patient." if risk_level == "High" else "Standard automated SMS."

    # 4. Save to MongoDB
    if db is not None:
        prediction_doc = {
            "timestamp": datetime.utcnow(),
            "raw_input": record.model_dump(),
            "cleaned_features": features.to_dict('records')[0],
            "risk_level": risk_level,
            "probability": float(probability),
            "recommendation": recommendation
        }
        db.predictions.insert_one(prediction_doc)

    return {
        "risk_level": risk_level,
        "probability": probability,
        "recommendation": recommendation
    }

@app.get("/history")
def get_history():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    # Fetch last 20 predictions, excluding MongoDB's internal _id
    records = list(db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
    return {"history": records}

@app.get("/stats")
def get_stats():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    pipeline = [
        { "$group": {
            "_id": None,
            "total_predictions": { "$sum": 1 },
            "high_risk_count": { "$sum": { "$cond": [{ "$eq": ["$risk_level", "High"] }, 1, 0] } },
            "low_risk_count": { "$sum": { "$cond": [{ "$eq": ["$risk_level", "Low"] }, 1, 0] } },
            "average_probability": { "$avg": "$probability" }
        }}
    ]
    
    result = list(db.predictions.aggregate(pipeline))
    if result:
        result[0].pop("_id", None)
        return result[0]
    return {"message": "No stats available yet."}