from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.export import router as export_router
from app.api.routes.trip import router as trip_router
from app.api.routes.weather import router as weather_router


app = FastAPI(
    title="WayCompass Backend",
    description="MVP backend for the intelligent travel assistant.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:80",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    """根路径，只用于确认服务已拉起。"""
    return {"message": "WayCompass backend is running."}


@app.get("/health")
def health_check() -> dict[str, str]:
    """健康探针，返回固定状态字段。"""
    return {"status": "ok"}


app.include_router(trip_router)
app.include_router(export_router)
app.include_router(weather_router)
