from fastapi import FastAPI, Response, status
from app.api.v1.api import api_router
from dotenv import load_dotenv
import os
import uvicorn


# Muat konfigurasi lingkungan dari file .env
load_dotenv()

def create_app() -> FastAPI:
    """
    Membuat instance FastAPI utama dan melakukan konfigurasi dasar aplikasi.
    """

    # Membaca variabel environment
    app_name = os.getenv("APP_NAME", "Predictive Maintenance Backend")
    app_env = os.getenv("APP_ENV", "development")

    app = FastAPI(
        title=app_name,
        description="API untuk deteksi anomali, prediksi mesin, data sensor, dan AI Copilot.",
        version="0.1.0",
    )

    # Daftarkan router API utama
    app.include_router(api_router, prefix="/api/v1")

    @app.get('/favicon.ico', include_in_schema=False)
    async def favicon():
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "status": "ok",
            "environment": app_env,
            "message": f"{app_name} is running!"
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto reload saat development
    )
