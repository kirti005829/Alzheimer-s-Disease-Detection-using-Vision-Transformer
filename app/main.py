from fastapi import FastAPI, UploadFile, File
from PIL import Image
import os
import shutil

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
from app.predictor import predict

app = FastAPI(
    title="Alzheimer Detection API",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "status": "Running",
        "model": "Vision Transformer",
        "classes": ["AD", "CI", "CN"]
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR,file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(file_path)

    result = predict(image)

    return result