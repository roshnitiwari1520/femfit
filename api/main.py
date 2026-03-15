from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import UserInput, FemFitResult
from ml.femfit_engine import (
    validate_inputs,
    bmr_standard,
    bmr_corrected,
    vo2_standard,
    vo2_corrected
)
from ml.phase_calculator import get_cycle_phase

app = FastAPI(
    title="FemFit API",
    description="Bias-corrected health metrics for women",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "FemFit API is running"}


@app.post("/calculate", response_model=FemFitResult)
def calculate(user: UserInput):
    try:
        validate_inputs(
            user.age,
            user.weight_kg,
            user.height_cm,
            user.resting_hr,
            user.hemoglobin_g_dl
        )

        phase = get_cycle_phase(user.last_period_date, user.cycle_length)

        std_bmr  = bmr_standard(user.weight_kg, user.height_cm, user.age)
        corr_bmr = bmr_corrected(user.weight_kg, user.height_cm, user.age, phase)
        std_vo2  = vo2_standard(user.resting_hr, user.age)
        corr_vo2 = vo2_corrected(user.resting_hr, user.age, user.hemoglobin_g_dl, phase)

        return FemFitResult(
            cycle_phase      = phase,
            bmr_standard     = std_bmr,
            bmr_corrected    = corr_bmr,
            bmr_delta        = round(corr_bmr - std_bmr, 2),
            vo2_standard     = std_vo2,
            vo2_corrected    = corr_vo2,
            vo2_delta        = round(corr_vo2 - std_vo2, 2),
            bmr_bias_percent = round((corr_bmr - std_bmr) / std_bmr * 100, 2),
            vo2_bias_percent = round((corr_vo2 - std_vo2) / std_vo2 * 100, 2),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))