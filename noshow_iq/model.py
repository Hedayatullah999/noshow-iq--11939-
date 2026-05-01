import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support

MODEL_PATH = "model.joblib"

def train(X_train, y_train):
    """Trains a Random Forest classifier, handling class imbalance."""
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(clf, MODEL_PATH)
    return clf

def evaluate(model, X_test, y_test):
    """Evaluates the model and returns precision, recall, and F1 for both classes."""
    y_pred = model.predict(X_test)
    
    # Get precision, recall, f1 for both classes
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    
    metrics = {
        "class_0_show": {"precision": precision[0], "recall": recall[0], "f1": f1[0]},
        "class_1_noshow": {"precision": precision[1], "recall": recall[1], "f1": f1[1]}
    }
    
    print(classification_report(y_test, y_pred))
    return metrics

def predict(features):
    """Loads the model and makes a prediction."""
    model = joblib.load(MODEL_PATH)
    # model.predict_proba returns [[prob_class_0, prob_class_1]]
    probability = model.predict_proba(features)[0][1] 
    risk_level = "High" if probability > 0.5 else "Low"
    return risk_level, probability