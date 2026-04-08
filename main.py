from fastapi import FastAPI
from api.routes import ping, ask, services

app = FastAPI(
    title="DarijaAssist API",
    version="1.0.0",
    description="Voice AI agent for Moroccan government services - API Contract v1.0"
)

app.include_router(ping.router)
app.include_router(ask.router)
app.include_router(services.router)