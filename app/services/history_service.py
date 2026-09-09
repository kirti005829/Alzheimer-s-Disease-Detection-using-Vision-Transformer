from sqlalchemy.orm import Session

from app.models import PredictionHistory


def save_prediction(
    db: Session,
    user_id: int,
    filename: str,
    prediction: str,
    confidence: str
):

    history = PredictionHistory(
        user_id=user_id,
        image_name=filename,
        prediction=prediction,
        confidence=confidence
    )

    db.add(history)
    db.commit()
    db.refresh(history)

    return history


def get_prediction_history(
    db: Session,
    user_id: int
):

    return (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == user_id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )