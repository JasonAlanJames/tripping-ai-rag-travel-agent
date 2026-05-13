from fastapi import APIRouter
from pydantic import BaseModel
from app.services.itinerary_service import generate_itinerary

router = APIRouter(prefix="/itinerary", tags=["Itinerary"])

class TripRequest(BaseModel):
    origin: str
    destination: str
    days: int
    budget: str

@router.post("/")
def create_itinerary(request: TripRequest):
    return generate_itinerary(request)