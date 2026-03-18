from pydantic import BaseModel, Field
from typing import Optional


class UserInput(BaseModel):
    age: int                    = Field(..., ge=18, le=50)
    weight_kg: float            = Field(..., ge=30, le=200)
    height_cm: float            = Field(..., ge=100, le=220)
    resting_hr: float           = Field(..., ge=40, le=120)
    hemoglobin_g_dl: float      = Field(..., ge=6.0, le=18.0)
    last_period_date: str
    cycle_length: Optional[int] = Field(default=28, ge=21, le=35)


class ZoneDetail(BaseModel):
    standard: str
    femfit: str
    delta: str


class WearableOutput(BaseModel):
    bmr: float
    calories_burned: float
    hr_zone_at_75pct: str
    max_hr: float


class FemFitOutput(BaseModel):
    bmr: float
    bmr_delta: float
    calories_burned: float
    calories_delta: float
    calories_bias_pct: float
    vo2: float
    vo2_lower: float
    vo2_upper: float
    vo2_delta: float
    max_hr: float
    effective_max_hr: float
    hr_zone_at_75pct: str
    hr_zones: dict
    fitness_label: str
    cycle_phase: str


class InsightMessage(BaseModel):
    category: str
    message: str


class FemFitResult(BaseModel):
    wearable_says:  WearableOutput
    femfit_says:    FemFitOutput
    insights:       list[InsightMessage]
    shap_explanation: list[str]