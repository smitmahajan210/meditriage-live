from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import joblib
import numpy as np
from PIL import Image, ImageOps
import io
import torchvision.transforms as transforms
from model_defs import SimpleNN, VGG13 
import json
import os

app = FastAPI(title="MediTriage AI API")

# --- 1. Load Models & Config ---
# We use relative paths so it works on Cloud and Local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_PATH = os.path.join(BASE_DIR, "../risk_model.pth")
OCR_MODEL_PATH = os.path.join(BASE_DIR, "../ocr_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "../scaler.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "../class_names.json")

# Load Tabular Model
risk_model = SimpleNN(input_features=7)
try:
    risk_model.load_state_dict(torch.load(RISK_MODEL_PATH, map_location=torch.device('cpu')))
    risk_model.eval()
except Exception as e:
    print(f"Warning: Risk Model not found: {e}")

# Load Scaler
try:
    scaler = joblib.load(SCALER_PATH)
except:
    print("Warning: Scaler not found")

# Load OCR Model
ocr_model = VGG13(num_classes=36)
try:
    ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=torch.device('cpu')))
    ocr_model.eval()
except Exception as e:
    print(f"Warning: OCR Model not found: {e}")

# Load Class Names
try:
    with open(CLASSES_PATH, 'r') as f:
        class_names_dict = json.load(f)
        class_names = list(class_names_dict.values()) if isinstance(class_names_dict, dict) else class_names_dict
except:
    class_names = [str(i) for i in range(36)]

# --- 2. Data Structures ---
class PatientData(BaseModel):
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float

# --- 3. SAFE Preprocessing (No OpenCV) ---
def process_image(image_bytes):
    # Open Image
    image = Image.open(io.BytesIO(image_bytes)).convert('L') # Grayscale
    
    # Simple Invert: If image is mostly white (paper), turn it black
    # This matches the training data (MNIST) which is white-on-black
    if np.mean(np.array(image)) > 127:
        image = ImageOps.invert(image)
    
    # Just Resize (No cropping/centering to avoid 'Black Screen' bugs)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return transform(image).unsqueeze(0)

# --- 4. API Endpoints ---
@app.get("/")
def home():
    return {"status": "Online"}

@app.post("/predict-risk")
def predict_risk(data: PatientData):
    input_data = np.array([[data.f1, data.f2, data.f3, data.f4, data.f5, data.f6, data.f7]])
    scaled_data = scaler.transform(input_data)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    
    with torch.no_grad():
        output = risk_model(input_tensor)
        probability = torch.sigmoid(output).item()
    
    return {
        "risk_probability": round(probability, 4),
        "prediction": "High Risk" if probability > 0.5 else "Low Risk"
    }

@app.post("/read-id")
async def read_id(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = process_image(image_bytes)
    
    with torch.no_grad():
        outputs = ocr_model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    return {"detected_id_char": class_names[predicted.item()]}