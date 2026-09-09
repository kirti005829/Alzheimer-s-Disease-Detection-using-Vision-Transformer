from fastapi import FastAPI
from app.routers import dashboard
from app.database import Base
from app.database import engine

from app.routers import auth
from app.routers import predict
from app.routers import history
from app.routers import gradcam

app = FastAPI(
    title="Alzheimer Detection API",
    version="1.0"
)

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(gradcam.router) 
app.include_router(dashboard.router)