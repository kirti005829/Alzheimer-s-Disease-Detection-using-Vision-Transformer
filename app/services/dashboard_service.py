from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import Prediction


def get_dashboard_statistics(
    db: Session,
    user_id: int
):

    predictions = (
        db.query(Prediction.prediction, func.count(Prediction.id))
        .filter(Prediction.user_id == user_id)
        .group_by(Prediction.prediction)
        .all()
    )

    stats = {
        "total_predictions": 0,
        "AD": 0,
        "CI": 0,
        "CN": 0
    }

    for prediction, count in predictions:

        stats[prediction] = count
        stats["total_predictions"] += count

    return stats