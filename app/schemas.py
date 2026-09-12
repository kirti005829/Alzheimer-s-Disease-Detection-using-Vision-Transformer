from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
class PredictionHistoryResponse(BaseModel):

    id: int

    filename: str

    prediction: str

    confidence: str

    created_at: datetime

    class Config:
        from_attributes = True
class PredictionResponse(BaseModel):

    id: int

    filename: str

    prediction: str

    confidence: float

    created_at: datetime

    class Config:

        from_attributes = True
class DashboardResponse(BaseModel):

    total_predictions: int

    AD: int

    CI: int

    CN: int