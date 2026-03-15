import pandas as pd
import numpy as np
from ml.phase_calculator import get_cycle_phase

# --- Constants ---
PHASE_BMR_MULTIPLIER = {
    "menstrual":  1.00,
    "follicular": 1.00,
    "ovulatory":  1.04,
    "luteal":     1.08
}

PHASE_VO2_MODIFIER = {
    "menstrual":  0.97,
    "follicular": 1.00,
    "ovulatory":  1.00,
    "luteal":     0.96
}

MALE_HB_BASELINE = 15.5  # g/dL


# --- Input Validation ---
def validate_inputs(age, weight_kg, height_cm, resting_hr, hemoglobin):
    if not (18 <= age <= 50):
        raise ValueError(f"Age {age} out of range. Must be 18–50.")
    if not (30 <= weight_kg <= 200):
        raise ValueError(f"Weight {weight_kg} out of range. Must be 30–200 kg.")
    if not (100 <= height_cm <= 220):
        raise ValueError(f"Height {height_cm} out of range. Must be 100–220 cm.")
    if not (40 <= resting_hr <= 120):
        raise ValueError(f"Resting HR {resting_hr} out of range. Must be 40–120 bpm.")
    if not (6.0 <= hemoglobin <= 18.0):
        raise ValueError(f"Hemoglobin {hemoglobin} out of range. Must be 6–18 g/dL.")


# --- BMR ---
def bmr_standard(weight_kg: float, height_cm: float, age: int) -> float:
    """Harris-Benedict female — biased baseline"""
    return round(655.1 + (9.563 * weight_kg) + (1.850 * height_cm) - (4.676 * age), 2)


def bmr_corrected(weight_kg: float, height_cm: float, age: int, cycle_phase: str) -> float:
    """Mifflin-St Jeor female + hormonal phase multiplier"""
    if cycle_phase not in PHASE_BMR_MULTIPLIER:
        raise ValueError(f"Invalid cycle phase: {cycle_phase}")
    base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    return round(base * PHASE_BMR_MULTIPLIER[cycle_phase], 2)


# --- VO2 Max ---
def vo2_standard(resting_hr: float, age: int) -> float:
    """Uth-Sørensen — male normed baseline"""
    max_hr = 220 - age
    return round(15 * (max_hr / resting_hr), 2)


def vo2_corrected(resting_hr: float, age: int, hemoglobin: float, cycle_phase: str) -> float:
    """Gulati max HR + personal Hb correction + phase modifier"""
    if cycle_phase not in PHASE_VO2_MODIFIER:
        raise ValueError(f"Invalid cycle phase: {cycle_phase}")
    true_max_hr = 206 - (0.88 * age)
    hb_correction = hemoglobin / MALE_HB_BASELINE
    base_vo2 = 15 * (true_max_hr / resting_hr)
    return round(base_vo2 * hb_correction * PHASE_VO2_MODIFIER[cycle_phase], 2)


# --- Vectorized (for full dataset in bias_analysis) ---
def run_on_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the full 1263-row dataframe.
    Returns same df with 4 new columns:
    bmr_standard, bmr_corrected, vo2_standard, vo2_corrected
    """
    required = {"age", "weight_kg", "height_cm", "resting_hr", "hemoglobin_g_dl", "cycle_phase"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df.copy()

    # BMR
    df["bmr_standard"] = (
        655.1 + (9.563 * df["weight_kg"]) +
        (1.850 * df["height_cm"]) - (4.676 * df["age"])
    ).round(2)

    df["bmr_multiplier"] = df["cycle_phase"].map(PHASE_BMR_MULTIPLIER)
    df["bmr_corrected"] = (
        ((10 * df["weight_kg"]) + (6.25 * df["height_cm"]) - (5 * df["age"]) - 161)
        * df["bmr_multiplier"]
    ).round(2)

    # VO2
    df["vo2_standard"] = (15 * ((220 - df["age"]) / df["resting_hr"])).round(2)

    df["true_max_hr"]    = 206 - (0.88 * df["age"])
    df["hb_correction"]  = df["hemoglobin_g_dl"] / MALE_HB_BASELINE
    df["vo2_modifier"]   = df["cycle_phase"].map(PHASE_VO2_MODIFIER)
    df["vo2_corrected"]  = (
        15 * (df["true_max_hr"] / df["resting_hr"])
        * df["hb_correction"]
        * df["vo2_modifier"]
    ).round(2)

    # Clean up temp columns
    df.drop(columns=["bmr_multiplier", "true_max_hr", "hb_correction", "vo2_modifier"],
            inplace=True)

    return df


# --- Single user test ---
if __name__ == "__main__":
    age, weight, height, resting_hr, hemoglobin = 25, 62.1, 158.4, 86.0, 12.5

    validate_inputs(age, weight, height, resting_hr, hemoglobin)
    phase = get_cycle_phase("2026-03-01", 28)

    std_bmr  = bmr_standard(weight, height, age)
    corr_bmr = bmr_corrected(weight, height, age, phase)
    std_vo2  = vo2_standard(resting_hr, age)
    corr_vo2 = vo2_corrected(resting_hr, age, hemoglobin, phase)

    print(f"Phase          : {phase}")
    print(f"BMR  standard  : {std_bmr} kcal")
    print(f"BMR  corrected : {corr_bmr} kcal  | delta: {round(corr_bmr - std_bmr, 2)}")
    print(f"VO2  standard  : {std_vo2} ml/kg/min")
    print(f"VO2  corrected : {corr_vo2} ml/kg/min  | delta: {round(corr_vo2 - std_vo2, 2)}")