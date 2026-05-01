import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.pkl')

FEATURES = [
    'Gender', 'Age', 'Scholarship', 'Hipertension',
    'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'days_in_advance'
]


def train(data_path: str = None):
    """
    Train the model on the dataset, handle class imbalance with SMOTE,
    save the model to model.pkl, and return evaluation metrics.
    """
    # --- Load dataset ---
    if data_path is None:
        # Look for the CSV in common locations
        candidates = [
            os.path.join(base_dir, '..', 'KaggleV2-May-2016.csv'),
            os.path.join(base_dir, 'KaggleV2-May-2016.csv'),
            'KaggleV2-May-2016.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset not found. Place KaggleV2-May-2016.csv in the project root."
        )

    df = pd.read_csv(data_path)

    # --- Fix column names (Kaggle dataset has messy names) ---
    df.columns = df.columns.str.strip()
    rename_map = {
        'Hipertension': 'Hipertension',   # keep as-is
        'No-show': 'No_show',
        'no-show': 'No_show',
        'No_Show': 'No_show',
    }
    df.rename(columns=rename_map, inplace=True)

    # Normalise the target column name robustly
    target_col = None
    for col in df.columns:
        if col.lower().replace('-', '_').replace(' ', '_') == 'no_show':
            target_col = col
            break
    if target_col is None:
        raise ValueError(f"Target column not found. Columns: {list(df.columns)}")

    # --- Clean data ---
    df = df[df['Age'] >= 0]          # remove negative ages
    df = df[df['Age'] <= 115]        # remove impossible ages

    # --- Engineer features ---
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')
    df['days_in_advance'] = (
        df['AppointmentDay'] - df['ScheduledDay']
    ).dt.days.clip(lower=0)

    df['Gender'] = (df['Gender'] == 'M').astype(int)

    # --- Target ---
    df['No_show'] = (df[target_col].str.strip().str.lower() == 'yes').astype(int)

    # --- Features / target split ---
    X = df[FEATURES].fillna(0)
    y = df['No_show']

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Handle class imbalance with SMOTE ---
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # --- Train classifier ---
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_bal, y_train_bal)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1]
    )

    metrics = {
        'timestamp': datetime.utcnow().isoformat(),
        'training_size': len(X_train_bal),
        'class_0': {
            'precision': round(float(precision[0]), 4),
            'recall':    round(float(recall[0]), 4),
            'f1':        round(float(f1[0]), 4),
        },
        'class_1': {
            'precision': round(float(precision[1]), 4),
            'recall':    round(float(recall[1]), 4),
            'f1':        round(float(f1[1]), 4),
        },
        'imbalance_technique': 'SMOTE + class_weight=balanced',
        'accuracy': round(float(report['accuracy']), 4),
    }

    # --- Save model ---
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    print(f"[train] Model saved to {model_path}")
    print(f"[train] Class 0 — P:{metrics['class_0']['precision']} R:{metrics['class_0']['recall']} F1:{metrics['class_0']['f1']}")
    print(f"[train] Class 1 — P:{metrics['class_1']['precision']} R:{metrics['class_1']['recall']} F1:{metrics['class_1']['f1']}")

    return metrics


def load_model():
    """Load the saved model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "model.pkl not found. Run train() first:\n"
            "  python -m noshow_iq.model"
        )
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict(data: dict):
    """
    Given a single appointment dict, return risk level, probability,
    and a recommendation string.
    """
    clf = load_model()

    # Build feature row
    gender = 1 if str(data.get('Gender', '')).strip().upper() == 'M' else 0

    # Compute days_in_advance if dates provided, else default to 0
    days_in_advance = int(data.get('days_in_advance', 0))
    if days_in_advance == 0:
        try:
            sched = pd.to_datetime(data.get('ScheduledDay'))
            appt  = pd.to_datetime(data.get('AppointmentDay'))
            days_in_advance = max(0, (appt - sched).days)
        except Exception:
            days_in_advance = 0

    row = {
        'Gender':        gender,
        'Age':           int(data.get('Age', 0)),
        'Scholarship':   int(data.get('Scholarship', 0)),
        'Hipertension':  int(data.get('Hipertension', 0)),
        'Diabetes':      int(data.get('Diabetes', 0)),
        'Alcoholism':    int(data.get('Alcoholism', 0)),
        'Handcap':       int(data.get('Handcap', 0)),
        'SMS_received':  int(data.get('SMS_received', 0)),
        'days_in_advance': days_in_advance,
    }

    input_df = pd.DataFrame([row])[FEATURES]
    probability = float(clf.predict_proba(input_df)[0][1])
    risk_level  = 'High' if probability >= 0.5 else 'Low'

    if risk_level == 'High':
        recommendation = (
            "Send an SMS reminder and consider overbooking this slot. "
            "Call the patient 24 hours before the appointment."
        )
    else:
        recommendation = (
            "Standard follow-up. No immediate action required."
        )

    return {
        'risk_level':     risk_level,
        'probability':    round(probability, 4),
        'recommendation': recommendation,
    }


def evaluate(data_path: str = None):
    """Re-evaluate the saved model on test data and print a report."""
    metrics = train(data_path)   # re-trains + evaluates
    return metrics


# Allow running directly: python -m noshow_iq.model
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    train(path)