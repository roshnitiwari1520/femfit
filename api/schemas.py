from pydantic import BaseModel, Field, validator
from typing import Optional


class UserInput(BaseModel):
    age: int = Field(..., ge=18, le=50)
    weight_kg: float = Field(..., ge=30, le=200)
    height_cm: float = Field(..., ge=100, le=220)
    resting_hr: float = Field(..., ge=40, le=120)
    hemoglobin_g_dl: float = Field(..., ge=6.0, le=18.0)
    last_period_date: str
    cycle_length: Optional[int] = Field(default=28, ge=21, le=35)


class FemFitResult(BaseModel):
    cycle_phase: str
    bmr_standard: float
    bmr_corrected: float
    bmr_delta: float
    vo2_standard: float
    vo2_corrected: float
    vo2_delta: float
    bmr_bias_percent: float
    vo2_bias_percent: float