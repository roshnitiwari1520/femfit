import pandas as pd
import numpy as np
import pickle
from ml.phase_calculator import get_cycle_phase

# ── Constants (kept for bias comparison only) ────────────────
PHASE_ENCODING = {
    "menstrual":  0,
    "follicular": 1,
    "ovulatory":  2,
    "luteal":     3
}

MALE_HB_BASELINE = 15.5


# ── Fitness Tier (AHA rule-based) ────────────────────────────
def get_fitness_tier(resting_hr: float) -> dict:
    if resting_hr < 60:
        return {"fitness_tier": 3, "fitness_label": "high"}
    elif resting_hr <= 75:
        return {"fitness_tier": 2, "fitness_label": "moderate"}
    else:
        return {"fitness_tier": 1, "fitness_label": "low"}


# ── Input Validation ─────────────────────────────────────────
def validate_inputs(age, weight_kg, height_cm, resting_hr, hemoglobin):
    if not (18 <= age <= 50):
        raise ValueError(f"Age {age} out of range. Must be 18–50.")
    if not (30 <= weight_kg <= 200):
        raise ValueError(f"Weight {weight_kg} out of range.")
    if not (100 <= height_cm <= 220):
        raise ValueError(f"Height {height_cm} out of range.")
    if not (40 <= resting_hr <= 120):
        raise ValueError(f"Resting HR {resting_hr} out of range.")
    if not (6.0 <= hemoglobin <= 18.0):
        raise ValueError(f"Hemoglobin {hemoglobin} out of range.")


# ── Standard (Biased) Formulas ───────────────────────────────
def bmr_standard(weight_kg, height_cm, age):
    """Harris-Benedict — male calibrated baseline"""
    return round(655.1 + (9.563*weight_kg) + (1.850*height_cm) - (4.676*age), 2)


def vo2_standard(resting_hr, age):
    """Uth-Sørensen — male normed baseline"""
    return round(15 * ((220 - age) / resting_hr), 2)


# ── FemFit Model Predictions ─────────────────────────────────
def _load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def bmr_femfit(weight_kg: float, height_cm: float,
               age: int, cycle_phase: str) -> float:
    """
    BMR predicted by model trained on female data.
    Learned coefficients replace Harris-Benedict constants.
    """
    model = _load_model("ml/models/bmr_model.pkl")
    phase_enc = PHASE_ENCODING[cycle_phase]
    X = pd.DataFrame([[weight_kg, height_cm, age, phase_enc]],
                     columns=["weight_kg", "height_cm", "age", "phase_encoded"])
    return round(float(model.predict(X)[0]), 2)


# ── Tiered Hb Correction ─────────────────────────────────────
def _hb_correction(hemoglobin: float) -> float:
    """
    Tiered hemoglobin correction.
    Captures the anemia cliff — not linear.
    """
    if hemoglobin > 14.0:
        return 0.97      # above normal female range
    elif hemoglobin >= 12.0:
        return 0.90      # normal female range
    elif hemoglobin >= 10.0:
        return 0.83      # mild anemia
    else:
        return 0.74      # clinical anemia — steep drop


# ── Fitness-Aware Phase Modifier ─────────────────────────────
PHASE_FITNESS_MODIFIER = {
    "menstrual":  {1: 0.96, 2: 0.97, 3: 0.98},
    "follicular": {1: 1.00, 2: 1.00, 3: 1.00},
    "ovulatory":  {1: 1.00, 2: 1.00, 3: 1.00},
    "luteal":     {1: 0.94, 2: 0.96, 3: 0.97},
}

def _vo2_confidence(predicted: float, hemoglobin: float,
                    cycle_phase: str, fitness_tier: int) -> tuple:
    """
    Dynamic confidence interval based on physiological uncertainty.
    Base SD: 3.5 ml/kg/min from published female VO2 measurement studies.
    Widens for anemia, luteal phase, low fitness.
    """
    base_sd = 3.5

    if hemoglobin < 10.0:
        hb_uncertainty = 1.4
    elif hemoglobin < 12.0:
        hb_uncertainty = 1.2
    else:
        hb_uncertainty = 1.0

    phase_uncertainty = {
        "menstrual":  1.15,
        "follicular": 1.00,
        "ovulatory":  1.00,
        "luteal":     1.20
    }[cycle_phase]

    fitness_uncertainty = {
        1: 1.15,
        2: 1.00,
        3: 0.90
    }[fitness_tier]

    sd = base_sd * hb_uncertainty * phase_uncertainty * fitness_uncertainty

    return (
        round(predicted - sd, 2),
        round(predicted + sd, 2)
    )

# ── VO2 FemFit (Final) ───────────────────────────────────────
def vo2_femfit(resting_hr: float, age: int,
               hemoglobin: float, cycle_phase: str) -> dict:
    """
    Female-specific VO2 Max correction.

    Corrections applied:
    1. Gulati female max HR (replaces 220-age)
    2. Learned female base coefficient 15.22 (replaces male 15.0)
    3. Tiered Hb correction (captures anemia cliff)
    4. Fitness-aware phase modifier (high fitness = smaller penalty)
    5. Compounding interaction (low Hb + luteal = extra penalty)
    """

    # Step 1 — Female max HR
    true_max_hr = 206 - (0.88 * age)

    # Step 2 — Base VO2 with learned female coefficient
    base_vo2 = 15.22 * (true_max_hr / resting_hr)

    # Step 3 — Tiered Hb correction
    hb_corr = _hb_correction(hemoglobin)

    # Step 4 — Fitness-aware phase modifier
    fitness_tier = get_fitness_tier(resting_hr)["fitness_tier"]
    phase_mod = PHASE_FITNESS_MODIFIER[cycle_phase][fitness_tier]

    # Step 5 — Compounding interaction
    if hemoglobin < 12.0 and cycle_phase == "luteal":
        interaction = 0.97
    else:
        interaction = 1.00

    predicted = round(base_vo2 * hb_corr * phase_mod * interaction, 2)

    # Remove all quantile model loading code
    # Replace with:
    lower, upper = _vo2_confidence(predicted, hemoglobin, cycle_phase, fitness_tier)

    return {
        "vo2_femfit":  predicted,
        "vo2_lower":   lower,
        "vo2_upper":   upper
    }


# ── Vectorized for bias analysis ─────────────────────────────
def run_on_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phase_encoded"] = df["cycle_phase"].map(PHASE_ENCODING)
    df["true_max_hr"]   = 206 - (0.88 * df["age"])
    df["hr_ratio"]      = df["true_max_hr"] / df["resting_hr"]
    df["fitness_tier"]  = df["resting_hr"].apply(
        lambda hr: 3 if hr < 60 else (2 if hr <= 75 else 1)
    )
    df["hb_x_phase"]    = df["hemoglobin_g_dl"] * df["phase_encoded"]
    df["hr_x_fitness"]  = df["hr_ratio"]         * df["fitness_tier"]
    df["hb_x_fitness"]  = df["hemoglobin_g_dl"]  * df["fitness_tier"]

    # Standard baselines
    df["bmr_standard"] = (
        655.1 + (9.563*df["weight_kg"]) +
        (1.850*df["height_cm"]) - (4.676*df["age"])
    ).round(2)
    df["vo2_standard"] = (15 * ((220 - df["age"]) / df["resting_hr"])).round(2)

    # FemFit model predictions
    bmr_model  = _load_model("ml/models/bmr_model.pkl")
    vo2_model  = _load_model("ml/models/vo2_model.pkl")

    bmr_features = ["weight_kg", "height_cm", "age", "phase_encoded"]
    vo2_features = [
        "hr_ratio", "hemoglobin_g_dl", "fitness_tier", "phase_encoded",
        "hb_x_phase", "hr_x_fitness", "hb_x_fitness"
    ]

    df["bmr_femfit"] = bmr_model.predict(df[bmr_features]).round(2)
    df["vo2_femfit"] = vo2_model.predict(df[vo2_features]).round(2)

    return df

# ── Heart Rate Zones ─────────────────────────────────────────
def hr_zones(age: int, resting_hr: int, cycle_phase: str) -> dict:
    """
    Female-corrected HR zones.

    Standard wearables use 220-age (male normed).
    We use Gulati formula + phase-aware intensity adjustment.

    Luteal/menstrual phase: zones shift down — same effort
    feels harder due to progesterone + temperature rise.
    """

    # Standard max HR (what wearables use)
    standard_max_hr = 220 - age

    # Female max HR (Gulati)
    female_max_hr = 206 - (0.88 * age)

    # Phase intensity modifier
    # Luteal/menstrual: perceived exertion higher → zones shift down
    PHASE_INTENSITY = {
        "menstrual":  0.97,
        "follicular": 1.00,
        "ovulatory":  1.00,
        "luteal":     0.96
    }
    phase_mod = PHASE_INTENSITY[cycle_phase]
    effective_max_hr = round(female_max_hr * phase_mod)

    def zones_from_max(max_hr: int) -> dict:
        return {
            "zone_1_rest":     (round(max_hr * 0.50), round(max_hr * 0.60)),
            "zone_2_fat_burn": (round(max_hr * 0.60), round(max_hr * 0.70)),
            "zone_3_cardio":   (round(max_hr * 0.70), round(max_hr * 0.80)),
            "zone_4_peak":     (round(max_hr * 0.80), round(max_hr * 0.90)),
            "zone_5_max":      (round(max_hr * 0.90), max_hr),
        }

    standard_zones = zones_from_max(standard_max_hr)
    femfit_zones   = zones_from_max(effective_max_hr)

    # Compute delta per zone
    deltas = {}
    for zone in standard_zones:
        std_low,  std_high  = standard_zones[zone]
        fem_low,  fem_high  = femfit_zones[zone]
        deltas[zone] = {
            "standard": f"{std_low}–{std_high} bpm",
            "femfit":   f"{fem_low}–{fem_high} bpm",
            "delta":    f"{fem_low - std_low} bpm"
        }

    return {
        "standard_max_hr":  standard_max_hr,
        "female_max_hr":    female_max_hr,
        "effective_max_hr": effective_max_hr,
        "cycle_phase":      cycle_phase,
        "zones":            deltas
    }


# ── Single user test ─────────────────────────────────────────
if __name__ == "__main__":
    age, weight, height, resting_hr, hemoglobin = 25, 62.1, 158.4, 86.0, 12.5

    validate_inputs(age, weight, height, resting_hr, hemoglobin)
    phase   = get_cycle_phase("2026-03-01", 28)
    fitness = get_fitness_tier(resting_hr)

    std_bmr  = bmr_standard(weight, height, age)
    pred_bmr = bmr_femfit(weight, height, age, phase)

    std_vo2  = vo2_standard(resting_hr, age)
    pred_vo2 = vo2_femfit(resting_hr, age, hemoglobin, phase)

    print(f"Phase          : {phase}")
    print(f"Fitness        : {fitness['fitness_label']}")
    print(f"\nBMR standard   : {std_bmr} kcal")
    print(f"BMR femfit     : {pred_bmr} kcal")
    print(f"BMR delta      : {round(pred_bmr - std_bmr, 2)} kcal")
    print(f"\nVO2 standard   : {std_vo2} ml/kg/min")
    print(f"VO2 femfit     : {pred_vo2['vo2_femfit']} ml/kg/min")
    print(f"VO2 range      : {pred_vo2['vo2_lower']} – {pred_vo2['vo2_upper']}")
    print(f"VO2 delta      : {round(pred_vo2['vo2_femfit'] - std_vo2, 2)} ml/kg/min")
    # HR Zones test
    zones = hr_zones(25, 86, phase)
    print(f"\nHR Zones (phase: {phase})")
    print(f"  Standard max HR : {zones['standard_max_hr']} bpm  (220 - age)")
    print(f"  Female max HR   : {zones['female_max_hr']} bpm  (Gulati)")
    print(f"  Effective max HR: {zones['effective_max_hr']} bpm  (phase-adjusted)")
    print(f"\n  {'Zone':<20} {'Standard':>15} {'FemFit':>15} {'Delta':>10}")
    print(f"  {'-'*60}")
    for zone, vals in zones["zones"].items():
        print(f"  {zone:<20} {vals['standard']:>15} {vals['femfit']:>15} {vals['delta']:>10}")