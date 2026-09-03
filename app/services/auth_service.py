from sqlalchemy.orm import Session

from app.models import User
from app.schemas import UserRegister
from app.security import (
    hash_password,
    verify_password,
    create_access_token
)


def register_user(user: UserRegister, db: Session):

    existing_user = (
        db.query(User)
        .filter(User.email == user.email)
        .first()
    )

    if existing_user:
        raise ValueError("Email already registered")

    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


def login_user(email: str, password: str, db: Session):

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if user is None:
        raise ValueError("Invalid Email")

    if not verify_password(
        password,
        user.hashed_password
    ):
        raise ValueError("Incorrect Password")

    token = create_access_token(
        {"sub": user.email}
    )

    return token