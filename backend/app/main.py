from fastapi import FastAPI
from app.routes.itinerary import router as itinerary_router

app = FastAPI(title="Tripping AI RAG Travel Agent")

app.include_router(itinerary_router)

@app.get("/")
def root():
    return {"status": "Tripping AI backend running"}