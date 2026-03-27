from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import os

app=FastAPI(
    title="Predictive Maintenance API",
    description="AI4I, Wind Turbine ve SCANIA modelleri için tahmin servisi",
    version="1.0.0"
)

BASE=os.path.dirname(__file__)
ai4i_model=joblib.load(os.path.join(BASE, "ai4i2020_model.pkl"))
turbine_model=joblib.load(os.path.join(BASE, "WindData.pkl"))
scania_model=joblib.load(os.path.join(BASE, "sciana_predictive_maintenance_model.pkl"))

class AI4IRequest(BaseModel):
    air_temperature:float # Kelvin—z.B.:298.1
    process_temperature:float # Kelvin—z.B.:308.6
    rotational_speed:float #rpm—z.B.:1551
    torque:float #Nm—z.B.:42.8
    tool_wear:float #min—z.B.:0
    type_L:Optional[int] = 0 #1=L Mschinetype
    type_M:Optional[int] = 0 # 1=M Maschinetype

class TurbineRequest(BaseModel):
    wind_speed:float #m/s z.B.: 5.3
    active_power:float #kW  z.B.: 380.0
    theoretical_power:float #KWh z.B.: 416.3
    wind_direction:float #Degree z.B.: 259.9
    hour:int #0-23
    month:int # 1-12

class ScaniaRequest(BaseModel):
    features: list[float]         # 170 Features

class PredictionResponse(BaseModel):
    failure:bool
    failure_probability:float
    confidence:str # Hoch / Mittle / tiefes Niveau
    recommendation:str # Deustsche Empfehlung
    model:str

def confidence_level(prob: float) -> str:
    if prob>=0.75:return "HIGH"
    if prob>=0.50:return "MEDIUM"
    return "LOW"

def recommendation(prob: float, model_type: str) -> str:
    if prob >= 0.75:
        msgs={
            "ai4i":"Sofortige Wartung erforderlich — Maschinenausfall droht",
            "turbine":"Sofortige Inspektion erforderlich — Leistungsabfall erkannt",
            "scania":"Fahrzeug sofort aus dem Betrieb nehmen — APS-Fehler kritisch",
        }
    elif prob >= 0.50:
        msgs={
            "ai4i":"Wartung in Kürze empfohlen — Anomalie erkannt",
            "turbine":"Überwachung empfohlen — Leistung unter Erwartung",
            "scania": "Diagnose empfohlen — Drucksystem prüfen",
        }
    else:
        msgs={
            "ai4i":"Normaler Betrieb — keine Maßnahmen erforderlich",
            "turbine":"Normaler Betrieb — Leistung im erwarteten Bereich",
            "scania":"Normaler Betrieb — APS-System in Ordnung",
        }
    return msgs.get(model_type, "Status unbekannt")
@app.get("/")
def root():
    return {
        "service":"Predictive Maintenance API",
        "endpoints":["/predict/machine", "/predict/turbine", "/predict/truck", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/machine", response_model=PredictionResponse)
def predict_machine(req: AI4IRequest):
    try:
        temp_diff=req.process_temperature - req.air_temperature
        power=req.rotational_speed * req.torque / 9549
        wear_torque=req.tool_wear * req.torque

        features=np.array([[
            req.air_temperature, req.process_temperature,
            req.rotational_speed, req.torque, req.tool_wear,
            req.type_L, req.type_M,
            temp_diff, power, wear_torque
        ]])

        prob=float(ai4i_model.predict_proba(features)[0][1])
        label=bool(ai4i_model.predict(features)[0])

        return PredictionResponse(
            failure=label,
            failure_probability=round(prob, 4),
            confidence=confidence_level(prob),
            recommendation=recommendation(prob, "ai4i"),
            model="RandomForest — AI4I 2020"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/turbine", response_model=PredictionResponse)
def predict_turbine(req:TurbineRequest):
    try:
        efficiency=min(req.active_power/(req.theoretical_power + 1e-6), 1.5)

        features=np.array([[
            req.wind_speed, req.theoretical_power, req.wind_direction,
            efficiency, req.hour, req.month
        ]])

        prob=float(turbine_model.predict_proba(features)[0][1])
        label=bool(turbine_model.predict(features)[0])

        return PredictionResponse(
            failure=label,
            failure_probability=round(prob, 4),
            confidence=confidence_level(prob),
            recommendation=recommendation(prob, "turbine"),
            model="RandomForest — Wind Turbine SCADA"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/truck", response_model=PredictionResponse)
def predict_truck(req: ScaniaRequest):
    try:
        if len(req.features)!= 170:
            raise HTTPException(
                status_code=422,
                detail=f"170 Features warten, {len(req.features)} wird abgesendet."
            )

        features = np.array([req.features])
        col_means=np.nanmean(features, axis=1, keepdims=True)
        nan_mask=np.isnan(features)
        features[nan_mask]=np.take(col_means, np.where(nan_mask)[0])

        prob=float(scania_model.predict_proba(features)[0][1])
        label=bool(scania_model.predict(features)[0])

        return PredictionResponse(
            failure=label,
            failure_probability=round(prob, 4),
            confidence=confidence_level(prob),
            recommendation=recommendation(prob, "scania"),
            model="RandomForest — SCANIA APS"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

