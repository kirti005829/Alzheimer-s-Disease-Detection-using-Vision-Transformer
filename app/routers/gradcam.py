from fastapi import APIRouter
router = APIRouter(
    prefix="/gradcam",
    tags=["Explainability"]
)