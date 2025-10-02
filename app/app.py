# app.py
import os
import io
import json
import traceback

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import numpy as np
import pandas as pd
import requests
import pickle
import torch
from torchvision import transforms
from PIL import Image

# local utilities (ensure these exist in utils/)
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9
import config

# -------------------- App setup --------------------
app = Flask(__name__)
CORS(app)  # allow cross-origin requests; adjust for production to restrict origins

# -------------------- Model paths & loading --------------------
# Ensure these paths match your repo layout
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.pth")
CROP_MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForest.pkl")
FERTILIZER_CSV_PATH = os.path.join(BASE_DIR, "Data", "fertilizer.csv")  # change if needed

# disease classes (keep as in your original)
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load disease model
try:
    disease_model = ResNet9(3, len(disease_classes))
    disease_model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location=torch.device('cpu')))
    disease_model.eval()
except Exception as e:
    disease_model = None
    print("ERROR loading disease model:", e)

# Load crop recommendation model (pickle)
try:
    with open(CROP_MODEL_PATH, "rb") as f:
        crop_recommendation_model = pickle.load(f)
except Exception as e:
    crop_recommendation_model = None
    print("ERROR loading crop recommendation model:", e)

# -------------------- utilities --------------------
def weather_fetch(city_name):
    """
    Fetch temperature (C) and humidity (%) for a city via OpenWeatherMap.
    Returns tuple (temperature, humidity) or None if city not found.
    """
    try:
        api_key = config.weather_api_key
        if not api_key:
            return None
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}appid={api_key}&q={city_name}"
        response = requests.get(complete_url, timeout=8)
        x = response.json()
        if x.get("cod") and str(x.get("cod")) != "404":
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2)
            humidity = y["humidity"]
            return temperature, humidity
        return None
    except Exception:
        return None

def predict_image_bytes(img_bytes, model=disease_model):
    """
    Transforms image bytes -> tensor -> model prediction label string.
    """
    if model is None:
        raise RuntimeError("disease model not loaded")

    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        yb = model(img_u)
        _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

def parse_int_field(d, key, default=None, required=False):
    """Utility: parse integer from dict/form"""
    val = d.get(key)
    if val is None:
        if required:
            raise ValueError(f"Missing required field: {key}")
        return default
    return int(val)

def parse_float_field(d, key, default=None, required=False):
    val = d.get(key)
    if val is None:
        if required:
            raise ValueError(f"Missing required field: {key}")
        return default
    return float(val)

# -------------------- Routes (JSON APIs) --------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "success": True,
        "data": {
            "message": "KrishiAI API running",
            "endpoints": {
                "POST /crop-predict": "json/form -> {nitrogen, phosphorous, pottasium, ph, rainfall, city}",
                "POST /fertilizer-predict": "json/form -> {cropname, nitrogen, phosphorous, pottasium}",
                "POST /disease-predict": "multipart/form-data -> file (image)"
            }
        }
    })

@app.route("/crop-predict", methods=["POST"])
def crop_prediction():
    """
    Accepts JSON or form-data fields:
    - nitrogen (N)
    - phosphorous (P)
    - pottasium (K)
    - ph
    - rainfall
    - city (for weather)
    Returns: recommended_crop and model details
    """
    try:
        payload = request.get_json(silent=True) or request.form
        # allow both field names for clarity
        N = parse_int_field(payload, "nitrogen", required=True)
        P = parse_int_field(payload, "phosphorous", required=True)
        K = parse_int_field(payload, "pottasium", required=True)
        ph = parse_float_field(payload, "ph", required=True)
        rainfall = parse_float_field(payload, "rainfall", required=True)
        city = payload.get("city")

        # fetch weather if city provided
        if city:
            wh = weather_fetch(city)
            if wh is None:
                return jsonify({"success": False, "error": "city not found or weather API key missing"}), 400
            temperature, humidity = wh
        else:
            # If city not provided, you may allow caller to pass temperature/humidity directly
            temperature = parse_float_field(payload, "temperature", default=25.0)
            humidity = parse_float_field(payload, "humidity", default=50.0)

        if crop_recommendation_model is None:
            return jsonify({"success": False, "error": "crop recommendation model not loaded"}), 500

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = str(my_prediction[0])

        return jsonify({"success": True, "data": {"recommended_crop": final_prediction}})

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": "internal server error", "details": str(e)}), 500

@app.route("/fertilizer-predict", methods=["POST"])
def fertilizer_prediction():
    """
    Accepts JSON or form-data:
    - cropname
    - nitrogen, phosphorous, pottasium
    Returns fertilizer recommendation string (from fertilizer_dic)
    """
    try:
        payload = request.get_json(silent=True) or request.form
        crop_name = payload.get("cropname")
        if not crop_name:
            return jsonify({"success": False, "error": "cropname is required"}), 400

        N = parse_int_field(payload, "nitrogen", required=True)
        P = parse_int_field(payload, "phosphorous", required=True)
        K = parse_int_field(payload, "pottasium", required=True)

        # load fertilizer csv (lazy load each request; small file so ok)
        if not os.path.exists(FERTILIZER_CSV_PATH):
            # try common fallback path
            alt_path = os.path.join(BASE_DIR, "app", "Data", "fertilizer.csv")
            if os.path.exists(alt_path):
                df = pd.read_csv(alt_path)
            else:
                return jsonify({"success": False, "error": f"fertilizer csv not found at {FERTILIZER_CSV_PATH}"}), 500
        else:
            df = pd.read_csv(FERTILIZER_CSV_PATH)

        # ensure crop exists in CSV (case-insensitive match)
        crop_rows = df[df['Crop'].str.lower() == crop_name.lower()]
        if crop_rows.empty:
            return jsonify({"success": False, "error": f"crop '{crop_name}' not found in fertilizer dataset"}), 400

        nr = crop_rows['N'].iloc[0]
        pr = crop_rows['P'].iloc[0]
        kr = crop_rows['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp_map = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp_map[max(temp_map.keys())]

        if max_value == "N":
            key = 'NHigh' if n < 0 else "Nlow"
        elif max_value == "P":
            key = 'PHigh' if p < 0 else "Plow"
        else:
            key = 'KHigh' if k < 0 else "Klow"

        recommendation_text = fertilizer_dic.get(key, "No recommendation found")
        return jsonify({"success": True, "data": {"recommendation_key": key, "recommendation": recommendation_text}})

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": "internal server error", "details": str(e)}), 500

@app.route("/disease-predict", methods=["POST"])
def disease_prediction():
    """
    Accepts multipart/form-data with 'file' (image).
    Returns disease label and human-friendly explanation from disease_dic.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "file (image) is required"}), 400

        file = request.files.get('file')
        if file.filename == "":
            return jsonify({"success": False, "error": "empty filename"}), 400

        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"success": False, "error": "empty file"}), 400

        if disease_model is None:
            return jsonify({"success": False, "error": "disease model not loaded"}), 500

        raw_pred = predict_image_bytes(img_bytes)
        explanation = disease_dic.get(raw_pred, "No description available")

        return jsonify({"success": True, "data": {"disease_label": raw_pred, "explanation": explanation}})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": "internal server error", "details": str(e)}), 500

# -------------------- run server --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # bind 0.0.0.0 for Cloud Run
    app.run(host="0.0.0.0", port=port, debug=False)
