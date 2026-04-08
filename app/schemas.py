from __future__ import annotations

from pydantic import BaseModel


class BreastInput(BaseModel):
    age: float
    mean_width: float
    max_width: float
    mean_height: float
    max_height: float
    mean_area: float
    max_area: float
    findings_count: float
    breast_density: float


class DiabetesInput(BaseModel):
    gender: int | str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: int | str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float


class HeartInput(BaseModel):
    high_bp: int
    high_chol: int
    stroke: int
    asthma_ever: int
    copd: int
    arthritis: int
    depression: int
    kidney_disease: int
    diabetes: int
    sex: int
    smoker: int
    phys_active: int
    age: float
    bmi: float


class LiverInput(BaseModel):
    Age: float
    Sex: int | str
    Ascites: int | str
    Hepatomegaly: int | str
    Spiders: int | str
    Edema: int | str
    Bilirubin: float
    Cholesterol: float
    Albumin: float
    Copper: float
    Alk_Phos: float
    SGOT: float
    Tryglicerides: float
    Platelets: float
    Prothrombin: float
