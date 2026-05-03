from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_PATH = "deepfake_model.h5"

# 🔥 Auto download model from Google Drive
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    os.system("pip install gdown")
   os.system("gdown https://drive.google.com/uc?id=1OCo3flOdyQjtTtWFuvARDjtnnQmnbmwj -O deepfake_model.h5")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded ✅")

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return "API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    processed = preprocess_image(img)

    pred = model.predict(processed)[0][0]

    if pred > 0.5:
        label = "Real"
        confidence = float(pred)
    else:
        label = "Fake"
        confidence = float(1 - pred)

    return jsonify({
        "label": label,
        "confidence": confidence
    })
