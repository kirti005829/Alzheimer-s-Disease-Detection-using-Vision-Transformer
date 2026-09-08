from sqlalchemy.orm import Session

from app.models import PredictionHistory


def save_prediction(
    db: Session,
    user_id: int,
    image_name: str,
    prediction: str,
    confidence: float
):

    history = PredictionHistory(
        user_id=user_id,
        image_name=image_name,
        prediction=prediction,
        confidence=str(confidence)
    )

    db.add(history)

    db.commit()

    db.refresh(history)

    return history