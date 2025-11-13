from fastapi import FastAPI
from app.api.v1.api import api_router
import uvicorn


def create_app() -> FastAPI:
    """
    Membuat instance FastAPI utama.
    Menyimpan konfigurasi dasar aplikasi dan route yang digunakan.
    """
    app = FastAPI(
        title="Predictive Maintenance Copilot API",
        description="API untuk deteksi anomali, prediksi, dan agent chatbot.",
        version="0.1.0"
    )

    # Register router versi 1
    app.include_router(api_router, prefix="/api/v1")

    # Root endpoint (health check)
    @app.get("/", tags=["Root"])
    def root():
        """
        Endpoint sederhana untuk mengecek apakah server berjalan.
        """
        return {"status": "ok", "message": "Predictive Maintenance BE Server is running!"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
