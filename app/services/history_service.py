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
        filename=filename,
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
def delete_prediction(db: Session, prediction_id: int, user_id: int):

    prediction = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.id == prediction_id,
            PredictionHistory.user_id == user_id
        )
        .first()
    )

    if prediction is None:
        return False

    db.delete(prediction)
    db.commit()

    return True