from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user

from app.schemas import PredictionHistoryResponse

from app.services.history_service import (
    get_prediction_history,
    delete_prediction
)

router = APIRouter(
    prefix="/history",
    tags=["History"]
)


@router.get(
    "/",
    response_model=list[PredictionHistoryResponse]
)
def history(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):

    return get_prediction_history(
        db,
        current_user.id
    )


@router.delete("/{prediction_id}")
def delete_history(
    prediction_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):

    deleted = delete_prediction(
        db,
        prediction_id,
        current_user.id
    )

    if not deleted:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    return {
        "message": "Prediction deleted successfully"
    }