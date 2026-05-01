import pandas as pd

def clean_data(df):
    """Clean the raw dataset and perform initial feature engineering."""
    df = df.copy()
    
    # 1. Fix typos in columns if they exist
    rename_map = {'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMSReceived'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 2. Convert date columns to datetime objects
    for col in ['ScheduledDay', 'AppointmentDay']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # 3. Filter negative ages ONLY if the column exists
    if 'Age' in df.columns:
        df = df[df['Age'] >= 0]

    # 4. Feature Engineering: Calculate days_in_advance
    if 'ScheduledDay' in df.columns and 'AppointmentDay' in df.columns:
        # FIX: Normalize to midnight then subtract to get the Timedelta
        # This keeps the .dt accessor available for .days
        scheduled = df['ScheduledDay'].dt.normalize()
        appointment = df['AppointmentDay'].dt.normalize()
        df['days_in_advance'] = (appointment - scheduled).dt.days
        
        # Ensure day_of_week is also present for the model
        df['day_of_week'] = df['AppointmentDay'].dt.dayofweek
        
    return df

def extract_features(df):
    """Select only the specific features required by the model for prediction."""
    features = [
        'Age', 'Scholarship', 'Hypertension', 'Diabetes', 
        'Alcoholism', 'Handicap', 'SMSReceived', 'days_in_advance', 'day_of_week'
    ]
    # Return only the columns that exist in the dataframe from our feature list
    return df[[col for col in features if col in df.columns]]