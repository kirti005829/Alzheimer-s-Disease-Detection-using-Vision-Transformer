from sqlalchemy.orm import Session
from app.models import Prediction


def save_prediction(
    db: Session,
    user_id: int,
    filename: str,
    prediction: str,
    confidence: float
):

    prediction_record = Prediction(
        user_id=user_id,
        filename=filename,
        prediction=prediction,
        confidence=confidence
    )

    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    return prediction_record
def get_prediction_history(
    db: Session,
    user_id: int
):

    return (
        db.query(Prediction)
        .filter(Prediction.user_id == user_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )


def delete_prediction(
    db: Session,
    prediction_id: int,
    user_id: int
):

    prediction = (
        db.query(Prediction)
        .filter(
            Prediction.id == prediction_id,
            Prediction.user_id == user_id
        )
        .first()
    )

    if prediction is None:

        return False

    db.delete(prediction)

    db.commit()

    return True