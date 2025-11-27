"""
Modul endpoint untuk API prediksi.

File ini TERINTEGRASI dengan model ML (.pkl) yang diberikan tim ML.
Endpoint akan:
- menerima data sensor,
- membentuk fitur sesuai feature_info.json,
- memanggil model,
- mengembalikan status mesin + probabilitas + pesan penjelasan.
"""

from fastapi import APIRouter, HTTPException
from app.schemas.prediction import PredictionInputSchema, PredictionOutputSchema
from pathlib import Path
from typing import Optional, Tuple

import sys
import json
import pickle
import joblib
import sklearn
import pandas as pd

# ---------------------------------------------------------
# 0. INISIALISASI ROUTER
# ---------------------------------------------------------

router = APIRouter()

# ---------------------------------------------------------
# 1. KONFIGURASI PATH & VARIABEL GLOBAL
# ---------------------------------------------------------

# path file ini: app/api/v1/endpoints/prediction.py
# parents[3] -> app/
BASE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "Model_prediksi_mesin.pkl"
FEATURE_INFO_PATH = MODELS_DIR / "feature_info.json"

model: Optional[object] = None
FEATURE_COLS: Optional[list[str]] = None


def _load_model_and_features() -> None:
    """
    Meload model ML (.pkl) dan informasi fitur dari feature_info.json.

    Dipanggil sekali saat modul di-import.
    Jika gagal, variabel global `model` akan None dan endpoint akan
    mengembalikan HTTP 500 ketika dipanggil.
    """
    global model, FEATURE_COLS

    print("[PREDICTION] --- Inisialisasi model ML ---")
    print("[PREDICTION] Python version:", sys.version.replace("\n", " "))
    print("[PREDICTION] scikit-learn version:", sklearn.__version__)
    print(f"[PREDICTION] MODEL_PATH: {MODEL_PATH}")
    print(f"[PREDICTION] FEATURE_INFO_PATH: {FEATURE_INFO_PATH}")

    # -------------------------------
    # 1) Load feature_info.json
    # -------------------------------
    try:
        with open(FEATURE_INFO_PATH, "r", encoding="utf-8") as f:
            feature_info = json.load(f)
        FEATURE_COLS = feature_info.get("feature_cols", [])
        if not FEATURE_COLS:
            print("[PREDICTION] PERINGATAN: feature_cols kosong di feature_info.json")
    except Exception as e:
        print(f"[PREDICTION] Gagal membaca feature_info.json: {e}")
        FEATURE_COLS = None

    # -------------------------------
    # 2) Load model (.pkl)
    # -------------------------------
    # Coba pakai joblib dulu (paling umum untuk model sklearn)
    try:
        print(f"[PREDICTION] Mencoba load model dengan joblib dari: {MODEL_PATH}")
        loaded_model = joblib.load(MODEL_PATH)
        model = loaded_model
        print("[PREDICTION] Model ML berhasil diload dengan joblib.")
        return
    except Exception as e:
        print(f"[PREDICTION] Gagal load model dengan joblib: {e}")

    # Fallback: coba pakai pickle biasa
    try:
        print(f"[PREDICTION] Mencoba load model dengan pickle dari: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            loaded_model = pickle.load(f)
        model = loaded_model
        print("[PREDICTION] Model ML berhasil diload dengan pickle.")
    except Exception as e:
        print(f"[PREDICTION] Gagal meload model ML dari {MODEL_PATH}: {e}")
        model = None


# Dipanggil sekali saat modul diimport
_load_model_and_features()

# ---------------------------------------------------------
# 2. ENDPOINT PREDIKSI
# ---------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictionOutputSchema,
    summary="Prediksi Kondisi Mesin",
    description="Mengembalikan hasil prediksi kondisi mesin berdasarkan data sensor yang diberikan.",
)
async def predict_failure(data: PredictionInputSchema) -> PredictionOutputSchema:
    """
    Endpoint utama untuk memproses data sensor dan menghasilkan prediksi kondisi mesin.
    """

    # Pastikan model dan fitur sudah terload dengan benar
    if model is None or not FEATURE_COLS:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model ML belum terkonfigurasi di server. "
                "Pastikan file model (.pkl) dan feature_info.json sudah sesuai "
                "dengan environment backend."
            ),
        )

    # -------------------------------
    # 1) Susun fitur sesuai feature_info.json
    # -------------------------------

    # feature_info.json (contoh):
    # ["Type", "Air temperature [K]", "Process temperature [K]",
    #  "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

    # Untuk sementara, 'Type' diisi default.
    # Jika nanti FE ingin menentukan L/M/H, field bisa ditambah di schema.
    default_type = "M"  # TODO: sesuaikan dengan kesepakatan tim ML (L/M/H)

    fitur_dict = {
        "Type": default_type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
    }

    try:
        print("[PREDICTION] FEATURE_COLS dari ML:", FEATURE_COLS)
        print("[PREDICTION] Kolom data yang diterima backend:", pd.DataFrame([fitur_dict]).columns.tolist())

        # Pastikan urutan kolom sesuai FEATURE_COLS dari tim ML
        df = pd.DataFrame([fitur_dict])[FEATURE_COLS]
    except Exception as e:
        print(f"[PREDICTION] Error menyusun DataFrame fitur: {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat menyiapkan data fitur untuk model ML.",
        )

    # -------------------------------
    # 2) Panggil model untuk prediksi
    # -------------------------------
    try:
        # Jika model mendukung probabilitas
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]

            # Asumsi kelas 1 = 'failure'
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    failure_idx = classes.index(1)
                    prob_failure = float(proba[failure_idx])
                else:
                    # fallback: pakai probabilitas terbesar
                    prob_failure = float(max(proba))
            else:
                prob_failure = float(max(proba))
        else:
            # Kalau tidak ada predict_proba, pakai predict biasa, probabilitas pseudo
            pred = model.predict(df)[0]
            prob_failure = 0.9 if pred == 1 else 0.1

    except Exception as e:
        print(f"[PREDICTION] Error saat inferensi model: {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat melakukan prediksi dengan model ML.",
        )

    # -------------------------------
    # 3) Interpretasi probabilitas
    # -------------------------------
    status, pesan = _interpret_probability(prob_failure)

    # -------------------------------
    # 4) Bentuk response
    # -------------------------------
    return PredictionOutputSchema(
        machine_status=status,
        probability=round(prob_failure, 4),
        message=pesan,
    )


# ---------------------------------------------------------
# 3. HELPER UNTUK INTERPRETASI PROBABILITAS
# ---------------------------------------------------------


def _interpret_probability(prob_failure: float) -> Tuple[str, str]:
    """
    Mengubah probabilitas failure menjadi status dan pesan penjelasan.
    Threshold dapat disesuaikan sesuai diskusi dengan tim ML.
    """

    if prob_failure < 0.3:
        status = "Normal"
        pesan = (
            "Mesin dalam kondisi normal. "
            "Probabilitas kegagalan rendah, tetap lakukan inspeksi rutin."
        )
    elif prob_failure < 0.7:
        status = "Warning"
        pesan = (
            "Waspada. Model mendeteksi peningkatan risiko kegagalan. "
            "Pertimbangkan penjadwalan maintenance preventif."
        )
    else:
        status = "Failure"
        pesan = (
            "Risiko kegagalan mesin tinggi. "
            "Disarankan dilakukan pemeriksaan segera dan penjadwalan perawatan."
        )

    return status, pesan
