# Import library yang dibutuhkan
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Inisialisasi aplikasi FastAPI dengan informasi dasar
# Title, deskripsi, dan versi ini akan muncul di dokumentasi Swagger (http://localhost:8000/docs)
app = FastAPI(
    title="Predictive Maintenance Copilot API",
    description="API untuk deteksi anomali, prediksi, dan agent chatbot.",
    version="0.1.0"
)

# Membuat model data (skema) untuk respons standar API
# BaseModel dari Pydantic digunakan agar struktur data lebih terkontrol
class StatusResponse(BaseModel):
    status: str   # Menunjukkan status permintaan (misalnya: "ok" atau "error")
    message: str  # Pesan penjelasan yang dikirimkan ke pengguna


# Endpoint utama (Root)
# GET / --> digunakan untuk memastikan server berjalan
# response_model memastikan struktur data sesuai dengan model StatusResponse
@app.get("/", response_model=StatusResponse, tags=["Root"])
def read_root():
    return {
        "status": "ok",
        "message": "Hello World - Predictive Maintenance BE Server is running!"
    }


# Endpoint Prediksi Maintenance
# POST /predict --> nantinya digunakan untuk mengirim data prediksi maintenance
# Saat ini masih dalam tahap pengembangan
@app.post("/predict", response_model=StatusResponse, tags=["Predictive Maintenance"])
def predict_maintenance():
    return {
        "status": "ok",
        "message": "Predictive maintenance endpoint is under construction."
    }

# Jalankan aplikasi
# Kode ini memastikan server hanya dijalankan ketika file ini dijalankan langsung,
# bukan saat diimpor sebagai modul di file lain.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
