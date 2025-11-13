"""
Modul endpoint untuk API prediksi.

File ini menyediakan logika sederhana (dummy model) 
untuk memperbaiki kondisi mesin berdasarkan data sensor.

"""

from fastapi import APIRouter
from app.schemas.prediction import PredictionInputSchema, PredictionOutputSchema


# Inisialisasi router API
router = APIRouter()

@router.post (
    "/predict", 
    response_model=PredictionOutputSchema,
    summary="Prediksi Kondisi Mesin",
    description="Mengembalikan hasil prediksi kondisi mesin berdasarkan data sensor yang diberikan.",   
)

async def predict_failure(data: PredictionInputSchema) -> PredictionOutputSchema:
    """
    Fungsi utama untuk memproses data sensor dan menghasilkan prediksi kondisi mesin.

    Argumen:
        data (PredictionInputSchema): Data masukan dari sensor mesin.

    Mengembalikan:
        PredictionOutputSchema: Hasil prediksi kondisi mesin beserta probabilitas dan pesan penjelasan. 
    
    """

    # Hitung skor resiko berdasarkan nilai sensor
    risk_score = _hitung_skor_risiko(data)

    # Tentukan hasil prediksi berdasarkan skor resiko
    status, probabilitas, pesan = _tentukan_status_mesin(risk_score)

    # Kembalikan hasil dalam format schema output
    return PredictionOutputSchema(
        machine_status=status,
        probability=probabilitas,
        message=pesan
    )

def _hitung_skor_risiko(data: PredictionInputSchema) -> int:
    """
    Menghitung skor resiko berdasarkan nilai data sensor yang diterima.
    Semakin tinggi nilai sensor tertenu, semakin besar potensi kerusakan.
    
    """
    skor = 0

    # Cek suhu udara atau suhu proses terlalu tinggi
    if data.air_temperature > 300 or data.process_temperature > 310:
        skor += 2

    # Cek kecepatan dan torsi tinggi
    if data.rotational_speed > 1500 or data.torque > 50:
        skor += 3

    # Cek tingkat keausan alat besar
    if data.tool_wear > 200:
        skor += 2

    return skor


def _tentukan_status_mesin(skor: int) -> tuple[str, float, str]:
    
    """
    Menentukan status mesin berdasarkan skor resiko yang dihitung.
    Mengembalikan tiga nilai: status, probrabilitas, dan pesan penjelasan.
    
    """
        
    if skor <= 2:
        return ("Normal", 0.1, "Mesin dalam kondisi normal dan stabil.")
    elif skor <= 4:
        return ("Warning", 0.6, "Waspada, kondisi mesin menunjukkan tanda keausan.")
    return ("Failure", 0.9, "Kemungkinan besar mesin akan mengalami kerusakan.")