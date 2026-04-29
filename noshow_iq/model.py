import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from noshow_iq.preprocess import load_data, clean_data

def train_model(data_path):
    # 1. Load and Clean
    df = load_data(data_path)
    df = clean_data(df)
    
    # 2. Select Features (Predictors)
    # Using simple numeric columns for this assignment
    features = ['Age', 'Hypertension', 'Diabetes', 'Alcoholism', 'SMSReceived', 'days_in_advance']
    X = df[features]
    
    # Target variable: Convert 'No'/'Yes' to 0/1
    y = df['NoShow'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Classifier
    print("Training the model... please wait.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    print("\n--- Model Performance ---")
    print(classification_report(y_test, predictions))
    
    return model

if __name__ == "__main__":
    data_path = "data/KaggleV2-May-2016.csv"
    try:
        trained_model = train_model(data_path)
        print("✅ Model training complete!")
    except Exception as e:
        print(f"❌ Training Error: {e}")