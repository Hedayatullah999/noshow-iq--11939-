import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the raw dataset."""
    return pd.read_csv(file_path)

def clean_data(df):
    """
    1. Rename misspelled columns.
    2. Fix invalid ages.
    3. Create feature: days_in_advance.
    """
    # Fix typos in the Kaggle dataset
    df = df.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap',
        'SMS_received': 'SMSReceived',
        'No-show': 'NoShow'
    })
    
    # Remove negative ages
    df = df[df['Age'] >= 0]
    
    # Convert dates and remove timezone info for calculation
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], utc=True).dt.tz_localize(None)
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], utc=True).dt.tz_localize(None)
    
    # Feature Engineering: Calculate days between scheduling and appointment
    df['days_in_advance'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days
    
    # Remove rows where appointment was set before the scheduled date
    df = df[df['days_in_advance'] >= 0]
    
    return df

if __name__ == "__main__":
    # Pointing to your local data folder
    try:
        raw_data_path = "data/KaggleV2-May-2016.csv"
        data = load_data(raw_data_path)
        cleaned_data = clean_data(data)
        print("✅ Preprocessing Successful!")
        print(f"Cleaned data shape: {cleaned_data.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")