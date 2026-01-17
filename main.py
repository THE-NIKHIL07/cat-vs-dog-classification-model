from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from tensorflow.keras.applications.efficientnet import preprocess_input

class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float

LABELS = {0: "Cat", 1: "Dog"}
MODEL_PATH = "best_cat_dog_model"
model = None

app = FastAPI(title="Cat vs Dog Classifier API", version="1.0")

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

IMG_SIZE = 224

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        return JSONResponse(status_code=500, content={"detail": "Model not loaded"})
    image_bytes = await file.read()
    input_array = preprocess_image(image_bytes)
    preds = model.predict(input_array)[0][0]
    prob_dog = float(preds)
    prob_cat = 1 - prob_dog
    label = "Dog" if prob_dog > 0.5 else "Cat"
    confidence = prob_dog if label == "Dog" else prob_cat
    return {"predicted_label": label, "confidence": confidence}

@app.get("/health")
def health():
    if model is None:
        return JSONResponse(status_code=500, content={"status": "error", "detail": "Model not loaded"})
    return {"status": "ok", "detail": "Model is loaded"}

@app.get("/")
def root():
    return {"message": "Welcome to the Cat vs Dog Prediction API"}
