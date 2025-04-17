from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil
import os
from PIL import Image

# Initialize FastAPI
app = FastAPI()

# Load trained Keras model
MODEL_PATH = "models/best_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Labels for prediction (Modify based on your dataset)
LABELS = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)  # Open the image
    image = image.resize((64, 64))  # Resize to (64, 64)
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)  # Predict using the model
    predicted_label = LABELS[np.argmax(prediction)]  # **Fixed indentation**

    return {"prediction": predicted_label, "confidence": float(np.max(prediction))}

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
