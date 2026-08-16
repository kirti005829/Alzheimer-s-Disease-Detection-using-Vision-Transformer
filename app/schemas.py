from pydantic import BaseModel
class Predictionresponse(BaseModel):
    prediction: str 
    confidence: float 
    