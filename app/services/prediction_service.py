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