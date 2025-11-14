"""
Modul pengelola router API versi 1 (v1)
File ini menggabungkan semua endpoint yang ada dalam versi API ini.

"""

from fastapi import APIRouter
from app.api.v1.endpoints import prediction

# inisialisasi router utama API versi 1
api_router = APIRouter()

# Daftarkan semua endpoint di router 1
api_router.include_router(
    prediction.router, 
    prefix="/prediction", 
    tags=["prediction"]
)
