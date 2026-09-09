from fastapi import APIRouter
from fastapi import Depends

from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user

from app.schemas import DashboardResponse

from app.services.dashboard_service import (
    get_dashboard_statistics
)

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"]
)


@router.get(
    "/",
    response_model=DashboardResponse
)
def dashboard(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):

    return get_dashboard_statistics(
        db,
        current_user.id
    )