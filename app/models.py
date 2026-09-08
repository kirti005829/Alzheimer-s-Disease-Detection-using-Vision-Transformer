from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class User(Base):

    __tablename__ = "users"

    id = Column(Integer,primary_key=True,index=True)

    username = Column(String,unique=True,nullable=False)

    email = Column(String,unique=True,nullable=False
)

    hashed_password = Column(
        String,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )
    predictions = relationship(
    "PredictionHistory",
    back_populates="user"
)
class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    image_name = Column(
        String,
        nullable=False
    )

    prediction = Column(
        String,
        nullable=False
    )

    confidence = Column(
        String,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    user = relationship(
    "User",
    back_populates="predictions"
)