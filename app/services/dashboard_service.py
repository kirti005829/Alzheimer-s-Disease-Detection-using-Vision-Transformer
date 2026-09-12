from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import PredictionHistory


def get_dashboard_statistics(
    db: Session,
    user_id: int
):

    predictions = (
        db.query(PredictionHistory.prediction, func.count(PredictionHistory.id))
        .filter(PredictionHistory.user_id == user_id)
        .group_by(PredictionHistory.prediction)
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