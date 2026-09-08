from fastapi import APIRouter, UploadFile, File
from app.predictor import predict
from PIL import Image
import shutil
import os
from fastapi import Depends
from app.dependencies import get_current_user
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.history_service import save_prediction

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def predict_image(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(file_path)

    result = predict(image)

    save_prediction(
    db=db,
    user_id=current_user.id,
    filename=file.filename,
    prediction=result["prediction"],
    confidence=result["confidence"]
)

    return result