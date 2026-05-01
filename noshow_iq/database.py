import datetime
from pymongo import MongoClient

# Connection string using your verified credentials from important4.PNG
MONGO_URI = "mongodb+srv://noshowiq:jkxIbjgydnTTsOAZ@cluster0.abtkgbd.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(MONGO_URI)
    # Verification ping
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
    
    db = client['noshow_db']
    
    # Required collections for Q4 Marks
    predictions_col = db['predictions']
    training_runs_col = db['training_runs']
    
except Exception as e:
    print(f"❌ Connection failed: {e}")