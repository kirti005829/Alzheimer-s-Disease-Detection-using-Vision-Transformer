from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models import User
from app.schemas import UserRegister
from sqlalchemy.exc import IntegrityError
from app.security import (
    hash_password,
    verify_password,
    create_access_token
)


def register_user(user: UserRegister, db: Session):

    existing_username = (
        db.query(User)
        .filter(User.username == user.username)
        .first()
)

    if existing_username:

        raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Username already exists."
    )
    existing_user = (
        db.query(User)
        .filter(User.email == user.email)
        .first()
    )

    if existing_user:

        raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Email already registered."
    )
    hashed_password = hash_password(
    user.password
)

    new_user = User(
    username=user.username,
    email=user.email,
    hashed_password=hashed_password
)

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

    except IntegrityError:
        db.rollback()
    
        raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Username or Email already exists."
    )

    return new_user


#def login_user(email: str, password: str, db: Session):
#
#    user = (
#        db.query(User)
#        .filter(User.email == email)
#        .first()
#    )
#
#    if user is None:
#        raise ValueError("Invalid Email")
#
#    if not verify_password(
#        password,
#        user.hashed_password
#    ):
#        raise ValueError("Incorrect Password")
#
#    token = create_access_token(
#        {"sub": user.email}
#    )
#
#    return token
def login_user(email: str, password: str, db: Session):

    print("=" * 50)
    print("Email received:", email)

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    print("User found:", user)
    print("=" * 50)

    if user is None:
        raise ValueError("Invalid Email")

    if not verify_password(password, user.hashed_password):
        raise ValueError("Incorrect Password")

    token = create_access_token({"sub": user.email})

    return token