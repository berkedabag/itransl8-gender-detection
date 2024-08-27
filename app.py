from xgboost import XGBClassifier
import config
import time
import pickle
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils import extract_features
import logging

# Initialize the FastAPI app
app = FastAPI()

# Initialize the model
MODEL = None

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the response model
class PredictionResponse(BaseModel):
    male: str
    female: str
    audio_file: str
    time_taken: str

# Function to make audio prediction
def audio_prediction(audio_file):
    try:
        logger.info(f"Extracting features from audio file: {audio_file}")
        feats = extract_features(
            audio_file, mel=True, mfcc=True, chroma=True, contrast=True)
        scaler = pickle.load(open(config.SCALAR_PATH, 'rb'))
        X = scaler.transform(feats.reshape(1, -1))
        pred = MODEL.predict_proba(X)
        return pred[0][1]
    except Exception as e:
        logger.error(f"Error during audio prediction: {e}")
        raise e

# Endpoint to get prediction
@app.get("/predict", response_model=PredictionResponse)
async def predict(audio_file: str):
    try:
        start_time = time.time()
        logger.info(f"Received prediction request for audio file: {audio_file}")
        
        male_prediction = audio_prediction(audio_file)
        female_prediction = 1 - male_prediction
        
        response = {
            "male": str(male_prediction),
            "female": str(female_prediction),
            "audio_file": str(audio_file),
            "time_taken": str(time.time() - start_time),
        }
        
        logger.info(f"Prediction response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        raise e

# Run the app
if __name__ == "__main__":
    import uvicorn
    MODEL = XGBClassifier()
    MODEL.load_model(config.MODEL_PATH)
    logger.info("Model loaded and server starting...")
    uvicorn.run(app, host="127.0.0.1", port=8005)
