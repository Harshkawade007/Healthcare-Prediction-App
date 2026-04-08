from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.schemas import BreastInput, DiabetesInput, HeartInput, LiverInput
from lung_disease_model import CLASS_NAMES, get_lung_inference_transforms, load_lung_weights


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
METADATA_DIR = BASE_DIR / "model_metadata"


app = FastAPI(
    title="AI-Driven Healthcare API",
    version="1.0.0",
    description="FastAPI backend for breast, diabetes, heart, liver, and lung disease models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_models() -> dict[str, Any]:
    return {
        "breast": joblib.load(MODELS_DIR / "breast_cancer.joblib"),
        "diabetes": joblib.load(MODELS_DIR / "diabetes.joblib"),
        "heart": joblib.load(MODELS_DIR / "heart_disease.joblib"),
        "liver": joblib.load(MODELS_DIR / "liver_disease.joblib"),
    }


@lru_cache(maxsize=1)
def get_lung_model() -> torch.nn.Module:
    return load_lung_weights(str(MODELS_DIR / "lung_disease.pth"), device="cpu")


@lru_cache(maxsize=1)
def get_lung_transform():
    return get_lung_inference_transforms()


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _binary_code(value: int) -> int:
    if value in (0, 1):
        return value
    if value in (1, 2, 3, 4):
        return 1 if value == 1 else 0
    return int(value)


def _normalize_gender(value: int | str) -> int:
    if isinstance(value, int):
        return value
    v = value.strip().lower()
    mapping = {"male": 0, "m": 0, "female": 1, "f": 1}
    if v not in mapping:
        raise HTTPException(status_code=422, detail=f"Unsupported gender value: {value}")
    return mapping[v]


def _normalize_smoking(value: int | str) -> int:
    if isinstance(value, int):
        return value
    v = value.strip().lower()
    mapping = {
        "no": 0,
        "never": 0,
        "no info": 0,
        "former": 1,
        "ever": 1,
        "not current": 1,
        "yes": 2,
        "current": 2,
    }
    if v not in mapping:
        raise HTTPException(status_code=422, detail=f"Unsupported smoking value: {value}")
    return mapping[v]


def _normalize_liver_categorical(value: int | str, mapping: dict[str, int]) -> int:
    if isinstance(value, int):
        return value
    v = value.strip().upper()
    if v not in mapping:
        raise HTTPException(status_code=422, detail=f"Unsupported category value: {value}")
    return mapping[v]


def _heart_age_cat(age: float) -> int:
    if age <= 40:
        return 0
    if age <= 60:
        return 1
    return 2


def _build_breast_features(payload: BreastInput) -> pd.DataFrame:
    p = payload.model_dump()
    width_ratio = _safe_div(p["max_width"], p["mean_width"])
    height_ratio = _safe_div(p["max_height"], p["mean_height"])
    area_ratio = _safe_div(p["max_area"], p["mean_area"])
    aspect_ratio = _safe_div(p["mean_width"], p["mean_height"])
    lesion_density_ratio = _safe_div(p["findings_count"], p["breast_density"])

    engineered = {
        "width_ratio": width_ratio,
        "age_x_aspect": p["age"] * aspect_ratio,
        "log_max_area": math.log1p(max(p["max_area"], 0.0)),
        "area_variability": p["max_area"] - p["mean_area"],
        "aspect_ratio": aspect_ratio,
        "age_x_density": p["age"] * lesion_density_ratio,
        "height_ratio": height_ratio,
        "density_x_mean_area": (p["breast_density"] * p["mean_area"]) / 1000.0,
        "elongation": max(width_ratio, height_ratio),
        "size_consistency": _safe_div(1.0, width_ratio + height_ratio + area_ratio),
    }
    columns = list(get_models()["breast"].feature_names_in_)
    return pd.DataFrame([[engineered[col] for col in columns]], columns=columns)


def _build_diabetes_features(payload: DiabetesInput) -> pd.DataFrame:
    p = payload.model_dump()
    p["gender"] = _normalize_gender(p["gender"])
    p["smoking_history"] = _normalize_smoking(p["smoking_history"])
    columns = list(get_models()["diabetes"].feature_names_in_)
    return pd.DataFrame([[p[col] for col in columns]], columns=columns)


def _build_heart_features(payload: HeartInput) -> pd.DataFrame:
    p = payload.model_dump()
    base = {
        "high_bp": _binary_code(p["high_bp"]),
        "high_chol": _binary_code(p["high_chol"]),
        "stroke": _binary_code(p["stroke"]),
        "asthma_ever": _binary_code(p["asthma_ever"]),
        "copd": _binary_code(p["copd"]),
        "arthritis": _binary_code(p["arthritis"]),
        "depression": _binary_code(p["depression"]),
        "kidney_disease": _binary_code(p["kidney_disease"]),
        "diabetes": int(p["diabetes"]),
        "sex": int(p["sex"]),
        "smoker": _binary_code(p["smoker"]),
        "phys_active": _binary_code(p["phys_active"]),
        "age": float(p["age"]),
        "bmi": float(p["bmi"]),
    }

    engineered = {
        "age_cat": _heart_age_cat(base["age"]),
        "age_x_bp": base["age"] * base["high_bp"],
        "age_x_diab": base["age"] * base["diabetes"],
        "chronic_count": (
            base["high_bp"]
            + base["diabetes"]
            + base["arthritis"]
            + base["stroke"]
            + base["copd"]
            + base["kidney_disease"]
            + base["high_chol"]
        ),
        "age_sq": base["age"] ** 2,
        "age_cu": base["age"] ** 3,
        "age_log": math.log1p(max(base["age"], 0.0)),
        "cv_risk_index": np.mean(
            [base["high_bp"], base["high_chol"], base["diabetes"], base["stroke"]]
        ),
        "resp_risk_index": np.mean([base["copd"], base["asthma_ever"]]),
        "burden_index": (
            base["high_bp"]
            + base["high_chol"]
            + base["diabetes"]
            + base["stroke"]
            + base["copd"]
            + base["arthritis"]
            + base["kidney_disease"]
            + base["depression"]
            + base["asthma_ever"]
        )
        / 9.0,
        "age_x_cv": 0.0,
        "age_bp_diab": base["age"] * base["high_bp"] * base["diabetes"],
        "bmi_x_age": base["bmi"] * base["age"],
        "bmi_log": math.log1p(max(base["bmi"], 0.0)),
        "bmi_sqrt": math.sqrt(max(base["bmi"], 0.0)),
    }
    engineered["age_x_cv"] = base["age"] * engineered["cv_risk_index"]

    row = {**base, **engineered}
    columns = list(get_models()["heart"].feature_names_)
    return pd.DataFrame([[row[col] for col in columns]], columns=columns)


def _build_liver_features(payload: LiverInput) -> pd.DataFrame:
    p = payload.model_dump()
    p["Sex"] = _normalize_liver_categorical(p["Sex"], {"F": 0, "M": 1})
    p["Ascites"] = _normalize_liver_categorical(p["Ascites"], {"N": 0, "Y": 1})
    p["Hepatomegaly"] = _normalize_liver_categorical(p["Hepatomegaly"], {"N": 0, "Y": 1})
    p["Spiders"] = _normalize_liver_categorical(p["Spiders"], {"N": 0, "Y": 1})
    p["Edema"] = _normalize_liver_categorical(p["Edema"], {"N": 0, "S": 1, "Y": 2})
    columns = list(get_models()["liver"].feature_names_in_)
    return pd.DataFrame([[p[col] for col in columns]], columns=columns)


def _prediction_response(model: Any, features: pd.DataFrame, class_labels: list[str] | None = None):
    pred = model.predict(features)
    predicted_class = int(pred[0])
    response = {
        "predicted_class": predicted_class,
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        response["probabilities"] = [float(x) for x in proba]
        if len(proba) > 1:
            response["positive_class_probability"] = float(proba[1])
    if class_labels:
        if 0 <= predicted_class < len(class_labels):
            response["predicted_label"] = class_labels[predicted_class]
        response["class_labels"] = class_labels
    return response


@app.get("/")
def root():
    return {
        "message": "Healthcare API is running.",
        "docs": "/docs",
        "models": ["breast", "diabetes", "heart", "liver", "lung"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata/{disease}")
def get_metadata(disease: str):
    metadata_file = METADATA_DIR / f"{disease}.json"
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail=f"Metadata not found for disease: {disease}")
    with metadata_file.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict/breast")
def predict_breast(payload: BreastInput):
    model = get_models()["breast"]
    features = _build_breast_features(payload)
    response = _prediction_response(
        model,
        features,
        class_labels=["lower-risk class", "higher-risk class"],
    )
    return {"disease": "breast_cancer", **response}


@app.post("/predict/diabetes")
def predict_diabetes(payload: DiabetesInput):
    model = get_models()["diabetes"]
    features = _build_diabetes_features(payload)
    response = _prediction_response(model, features, class_labels=["no diabetes", "diabetes"])
    return {"disease": "diabetes", **response}


@app.post("/predict/heart")
def predict_heart(payload: HeartInput):
    model = get_models()["heart"]
    features = _build_heart_features(payload)
    response = _prediction_response(
        model,
        features,
        class_labels=["no heart disease", "heart disease"],
    )
    return {"disease": "heart_disease", **response}


@app.post("/predict/liver")
def predict_liver(payload: LiverInput):
    model = get_models()["liver"]
    features = _build_liver_features(payload)
    response = _prediction_response(
        model,
        features,
        class_labels=["stage_1", "stage_2", "stage_3"],
    )
    return {"disease": "liver_disease", **response}


@app.post("/predict/lung")
async def predict_lung(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Could not read image file.") from exc

    model = get_lung_model()
    transform = get_lung_transform()
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))

    return {
        "disease": "lung_disease",
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "class_labels": CLASS_NAMES,
        "probabilities": [float(x) for x in probs],
    }
