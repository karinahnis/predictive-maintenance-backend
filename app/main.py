from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Predictive Maintenance Copilot API",
    description="API untuk deteksi anomali, prediksi, dan agent chatbot.",
    version="0.1.0"
)

class StatusResponse(BaseModel):
    status: str
    message: str
# root route
@app.get("/", response_model=StatusResponse, tags=["Root"])
def read_root():
    return {
        "status": "ok", 
        "message": "Hello World - Predictive Maintenance BE Server is running!"
    }

# route /predict for predictive maintenance
@app.post("/predict", response_model=StatusResponse, tags=["Predictive Maintenance"])
def predict_maintenance():
    return {
        "status": "ok", 
        "message": "Predictive maintenance endpoint is under construction."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)     