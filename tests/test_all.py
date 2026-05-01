import pandas as pd
from fastapi.testclient import TestClient
from noshow_iq.preprocess import clean_data, extract_features
from noshow_iq.api import app

client = TestClient(app)

def test_health_endpoint():
    """Test 1: Check if API is healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_preprocess_renames_columns():
    """Test 2: Check if typos in dataset are fixed."""
    df = pd.DataFrame({'Hipertension': [1], 'Handcap': [0]})
    cleaned_df = clean_data(df)
    assert 'Hypertension' in cleaned_df.columns
    assert 'Hipertension' not in cleaned_df.columns

def test_preprocess_drops_negative_age():
    """Test 3: Ensure negative ages are removed."""
    df = pd.DataFrame({'Age': [25, -1, 40], 'ScheduledDay': ['2016-04-29T18:38:08Z']*3, 'AppointmentDay': ['2016-04-29T00:00:00Z']*3})
    cleaned_df = clean_data(df)
    assert len(cleaned_df) == 2

def test_preprocess_calculates_days_in_advance():
    """Test 4: Verify days_in_advance logic."""
    df = pd.DataFrame({
        'Age': [25],
        'ScheduledDay': ['2016-04-20T18:38:08Z'],
        'AppointmentDay': ['2016-04-25T00:00:00Z']
    })
    cleaned_df = clean_data(df)
    assert cleaned_df['days_in_advance'].iloc[0] == 5

def test_extract_features_columns():
    """Test 5: Ensure correct features are extracted for the model."""
    df = pd.DataFrame({
        'Age': [25], 'Scholarship': [0], 'Hypertension': [1], 'Diabetes': [0],
        'Alcoholism': [0], 'Handicap': [0], 'SMSReceived': [1], 
        'days_in_advance': [5], 'day_of_week': [0], 'ExtraCol': [99]
    })
    features = extract_features(df)
    assert 'ExtraCol' not in features.columns
    assert len(features.columns) == 9

def test_predict_endpoint_without_model():
    """Test 6: POST /predict fails gracefully if model.joblib is missing."""
    # Assuming model isn't trained during pure unit tests
    payload = {
        "PatientId": 12345, "AppointmentID": 56789, "Gender": "F",
        "ScheduledDay": "2016-04-29T18:38:08Z", "AppointmentDay": "2016-04-29T00:00:00Z",
        "Age": 30, "Neighbourhood": "JARDIM DA PENHA", "Scholarship": 0,
        "Hipertension": 0, "Diabetes": 0, "Alcoholism": 0, "Handcap": 0, "SMS_received": 0
    }
    response = client.post("/predict", json=payload)
    # Should throw 500 because we haven't run train() to generate model.joblib in the test environment yet
    assert response.status_code == 500