from fastapi import FastAPI
from app.routes.itinerary import router as itinerary_router
from app.routes.chat import router as chat_router

app = FastAPI(title="Tripping AI RAG Travel Agent")

app.include_router(itinerary_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {"status": "Tripping AI backend running"}