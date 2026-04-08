from fastapi import APIRouter
from models.response_models import ServicesResponse, ServiceItem

router = APIRouter()

@router.get("/services", response_model=ServicesResponse)
def get_services():
    """
    Returns available topics for icon shortcuts on home screen.
    Exact data from API contract.
    """
    return ServicesResponse(
        services=[
            ServiceItem(
                id="cnss",
                label_darija="الصندوق الوطني للضمان الاجتماعي",
                label_latin="CNSS"
            ),
            ServiceItem(
                id="amo",
                label_darija="التأمين الإجباري عن المرض",
                label_latin="AMO"
            ),
            ServiceItem(
                id="cin",
                label_darija="البطاقة الوطنية",
                label_latin="CIN"
            ),
            ServiceItem(
                id="moqawala",
                label_darija="المقاولة",
                label_latin="Moqawala"
            )
        ]
    )