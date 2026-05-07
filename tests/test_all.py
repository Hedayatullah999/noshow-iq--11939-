import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient


def test_health_endpoint():
    from noshow_iq.api import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_preprocess_drops_negative_age():
    from noshow_iq.preprocess import clean_data
    df = pd.DataFrame({
        'Age': [25, -1, 40],
        'ScheduledDay': ['2016-04-29T18:38:08Z'] * 3,
        'AppointmentDay': ['2016-04-29T00:00:00Z'] * 3
    })
    cleaned_df = clean_data(df)
    assert len(cleaned_df) == 2


def test_preprocess_calculates_days_in_advance():
    from noshow_iq.preprocess import clean_data
    df = pd.DataFrame({
        'Age': [25],
        'ScheduledDay': ['2016-04-20T18:38:08Z'],
        'AppointmentDay': ['2016-04-25T00:00:00Z']
    })
    cleaned_df = clean_data(df)
    assert 'days_in_advance' in cleaned_df.columns
    assert cleaned_df['days_in_advance'].iloc[0] == 5


def test_extract_features_returns_dataframe():
    from noshow_iq.preprocess import clean_data, extract_features
    df = pd.DataFrame({
        'Age': [25],
        'ScheduledDay': ['2016-04-20T18:38:08Z'],
        'AppointmentDay': ['2016-04-25T00:00:00Z'],
        'Gender': ['F'],
        'Scholarship': [0],
        'Hipertension': [0],
        'Diabetes': [0],
        'Alcoholism': [0],
        'Handcap': [0],
        'SMS_received': [1],
    })
    cleaned = clean_data(df)
    features = extract_features(cleaned)
    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1


def test_clean_data_handles_minimal_input():
    from noshow_iq.preprocess import clean_data
    df = pd.DataFrame({
        'Age': [30],
        'ScheduledDay': ['2016-04-20T18:38:08Z'],
        'AppointmentDay': ['2016-04-25T00:00:00Z']
    })
    cleaned = clean_data(df)
    assert cleaned is not None
    assert len(cleaned) == 1


def test_predict_endpoint_without_model():
    with patch('noshow_iq.model.load_model', side_effect=FileNotFoundError("model.pkl not found")):
        from noshow_iq.api import app
        client = TestClient(app)
        payload = {
            "PatientId": 12345,
            "AppointmentID": 56789,
            "Gender": "F",
            "ScheduledDay": "2016-04-29T18:38:08Z",
            "AppointmentDay": "2016-04-29T00:00:00Z",
            "Age": 30,
            "Neighbourhood": "JARDIM DA PENHA",
            "Scholarship": 0,
            "Hipertension": 0,
            "Diabetes": 0,
            "Alcoholism": 0,
            "Handcap": 0,
            "SMS_received": 0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code in [500, 200]

