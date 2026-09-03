from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas import (
    UserRegister,
    UserLogin,
    TokenResponse
)

from app.services.auth_service import (
    register_user,
    login_user
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)
@router.post("/register")
def register(
    user: UserRegister,
    db: Session = Depends(get_db)
):
    try:
        register_user(user, db)

        return {
            "message": "User registered successfully"
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
@router.post(
    "/login",
    response_model=TokenResponse
)
def login(
    user: UserLogin,
    db: Session = Depends(get_db)
):

    token = login_user(

        user.email,

        user.password,

        db

    )

    if token is None:

        raise HTTPException(

            status_code=401,

            detail="Invalid credentials"

        )

    return {

        "access_token": token,

        "token_type": "bearer"

    }