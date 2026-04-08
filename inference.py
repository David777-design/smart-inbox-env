import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
from typing import Optional

app = FastAPI(title="Smart Inbox Inference Service", version="1.0.0")

# Model loading
model = None
try:
    # Try to load scikit-learn model
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Loaded scikit-learn model from model.pkl")
    # Try to load PyTorch model
    elif os.path.exists('model.pt'):
        import torch
        model = torch.load('model.pt')
        model.eval()
        print("Loaded PyTorch model from model.pt")
    # Try to load Hugging Face model
    else:
        from transformers import pipeline
        model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        print("Loaded default Hugging Face model")
except Exception as e:
    print(f"Could not load model: {e}")
    model = None

class PredictionRequest(BaseModel):
    message: str
    sender: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict message priority"""
    if model is None:
        # Fallback prediction based on keywords
        text = request.message.lower()
        if any(word in text for word in ['urgent', 'deadline', 'immediately', 'asap', 'critical']):
            prediction = "high"
        elif any(word in text for word in ['meeting', 'report', 'review', 'follow up']):
            prediction = "medium"
        else:
            prediction = "low"
        return PredictionResponse(prediction=prediction)

    try:
        # For scikit-learn model (assuming text classification)
        if hasattr(model, 'predict'):
            # Simple feature extraction (in real scenario, use proper preprocessing)
            features = [len(request.message), request.message.count('!'), request.message.count('?')]
            prediction = model.predict([features])[0]
            return PredictionResponse(prediction=str(prediction))

        # For Hugging Face pipeline
        elif hasattr(model, 'predict') or callable(model):
            result = model(request.message)
            if isinstance(result, list) and result:
                prediction = result[0]['label']
                confidence = result[0]['score']
                # Map to our priority levels
                if 'POSITIVE' in prediction.upper():
                    prediction = "high"
                elif 'NEGATIVE' in prediction.upper():
                    prediction = "low"
                else:
                    prediction = "medium"
                return PredictionResponse(prediction=prediction, confidence=confidence)

        # For PyTorch model (assuming custom model)
        else:
            # This would need to be adapted based on actual model architecture
            # For now, return fallback
            return PredictionResponse(prediction="medium")

    except Exception as e:
        print(f"Prediction error: {e}")
        return PredictionResponse(prediction="low")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)