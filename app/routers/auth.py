from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    UserRegister,
    UserLogin,
    UserResponse,
    Token
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
    response_model=Token
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
@router.post(
    "/token",
    response_model=Token,
    include_in_schema=False
)
def login_swagger(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    token = login_user(
        form_data.username,
        form_data.password,
        db
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }