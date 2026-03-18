from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import UserInput, FemFitResult
from ml.femfit_engine import (
    validate_inputs, bmr_standard, bmr_femfit,
    vo2_standard, vo2_femfit, get_fitness_tier, hr_zones
)
from ml.phase_calculator import get_cycle_phase
from ml.calorie_model import predict_calories, explain_prediction

app = FastAPI(
    title="FemFit API",
    description="Bias-corrected health metrics for women",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_zone_at_effort(exercise_hr: float, max_hr: float) -> str:
    pct = exercise_hr / max_hr
    if pct < 0.60:   return "Rest"
    elif pct < 0.70: return "Fat Burn"
    elif pct < 0.80: return "Cardio"
    elif pct < 0.90: return "Peak"
    else:            return "Maximum"


def generate_insights(phase: str, fitness_label: str,
                      bmr_delta: float, cal_delta: float,
                      std_zone: str, fem_zone: str,
                      hemoglobin: float) -> list:
    insights = []

    # BMR insight
    if abs(bmr_delta) >= 50:
        direction = "underestimating" if bmr_delta < 0 else "overestimating"
        insights.append({
            "category": "BMR",
            "message": f"Your wearable is {direction} your resting "
                       f"calorie needs by {abs(bmr_delta):.0f} kcal daily. "
                       f"Over a month that's {abs(bmr_delta)*30:.0f} kcal."
        })

    # Phase insight
    phase_messages = {
        "luteal": (
            "You are in luteal phase. Progesterone raises your core "
            "temperature and ventilation rate — the same workout feels "
            "harder. Your calorie burn and performance zones are adjusted."
        ),
        "menstrual": (
            "You are in menstrual phase. Iron loss can reduce oxygen "
            "carrying capacity. FemFit accounts for this in your VO2 "
            "and calorie estimates."
        ),
        "follicular": (
            "You are in follicular phase — your peak performance window. "
            "Estrogen is rising, energy is highest. Great time for "
            "high intensity training."
        ),
        "ovulatory": (
            "You are in ovulatory phase. Near-peak performance. "
            "Strength and endurance are at their best right now."
        )
    }
    insights.append({
        "category": "Cycle Phase",
        "message": phase_messages[phase]
    })

    # HR Zone insight
    if std_zone != fem_zone:
        insights.append({
            "category": "HR Zone",
            "message": f"Your wearable thinks you are in {std_zone} zone. "
                       f"FemFit says you are actually in {fem_zone} zone. "
                       f"You may be working harder than you think."
        })

    # Hemoglobin insight
    if hemoglobin < 12.0:
        insights.append({
            "category": "Hemoglobin",
            "message": f"Your hemoglobin ({hemoglobin} g/dL) is below "
                       f"normal female range (12–15.5 g/dL). This limits "
                       f"oxygen delivery and reduces aerobic capacity. "
                       f"Consider consulting a healthcare provider."
        })

    # Calorie insight
    if abs(cal_delta) >= 15:
        insights.append({
            "category": "Calorie Burn",
            "message": f"Your wearable overcounts calorie burn by "
                       f"{abs(cal_delta):.0f} kcal per 30-min session. "
                       f"That's {abs(cal_delta)*30:.0f} kcal per month."
        })

    return insights


@app.get("/")
def root():
    return {"status": "FemFit API v2.0 running"}


@app.post("/calculate", response_model=FemFitResult)
def calculate(user: UserInput):
    try:
        validate_inputs(
            user.age, user.weight_kg, user.height_cm,
            user.resting_hr, user.hemoglobin_g_dl
        )

        # Phase + fitness
        phase   = get_cycle_phase(user.last_period_date, user.cycle_length)
        fitness = get_fitness_tier(user.resting_hr)

        # BMR
        std_bmr  = bmr_standard(user.weight_kg, user.height_cm, user.age)
        pred_bmr = bmr_femfit(user.weight_kg, user.height_cm,
                              user.age, phase)
        bmr_delta = round(pred_bmr - std_bmr, 2)

        # VO2
        std_vo2  = vo2_standard(user.resting_hr, user.age)
        pred_vo2 = vo2_femfit(user.resting_hr, user.age,
                              user.hemoglobin_g_dl, phase)
        vo2_delta = round(pred_vo2["vo2_femfit"] - std_vo2, 2)

        # Calories
        cal_result = predict_calories(
            user.weight_kg, user.age, user.resting_hr,
            user.hemoglobin_g_dl, phase
        )

        # HR Zones
        zones = hr_zones(user.age, int(user.resting_hr), phase)

        # Exercise HR at 75% effort
        exercise_hr = user.resting_hr + (user.resting_hr * 0.75)
        std_zone = get_zone_at_effort(exercise_hr, zones["standard_max_hr"])
        fem_zone = get_zone_at_effort(exercise_hr, zones["effective_max_hr"])

        # SHAP explanation
        shap_exp = explain_prediction(
            user.weight_kg, user.age, user.resting_hr,
            user.hemoglobin_g_dl, phase
        )

        # Insights
        insights = generate_insights(
            phase, fitness["fitness_label"],
            bmr_delta, cal_result["calories_delta"],
            std_zone, fem_zone, user.hemoglobin_g_dl
        )

        return FemFitResult(
            wearable_says=dict(
                bmr              = std_bmr,
                calories_burned  = cal_result["calories_standard"],
                hr_zone_at_75pct = std_zone,
                max_hr           = float(zones["standard_max_hr"])
            ),
            femfit_says=dict(
                bmr              = pred_bmr,
                bmr_delta        = bmr_delta,
                calories_burned  = cal_result["calories_femfit"],
                calories_delta   = cal_result["calories_delta"],
                calories_bias_pct= cal_result["calories_bias_pct"],
                vo2              = pred_vo2["vo2_femfit"],
                vo2_lower        = pred_vo2["vo2_lower"],
                vo2_upper        = pred_vo2["vo2_upper"],
                vo2_delta        = vo2_delta,
                max_hr           = float(zones["female_max_hr"]),
                effective_max_hr = float(zones["effective_max_hr"]),
                hr_zone_at_75pct = fem_zone,
                hr_zones         = zones["zones"],
                fitness_label    = fitness["fitness_label"],
                cycle_phase      = phase
            ),
            insights        = insights,
            shap_explanation= shap_exp["explanation"]
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))