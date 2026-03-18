import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ── Phase encoding ──────────────────────────────────────────
PHASE_ENCODING = {
    "menstrual":  0,
    "follicular": 1,
    "ovulatory":  2,
    "luteal":     3
}

def encode_phase(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phase_encoded"] = df["cycle_phase"].map(PHASE_ENCODING)
    return df


# ── BMR Model ────────────────────────────────────────────────
def train_bmr_model(df: pd.DataFrame):
    """
    Learns female-specific BMR coefficients from data.
    Features: weight, height, age, phase
    Target: bmr_measured (real or simulated benchmark)
    """
    df = encode_phase(df)

    features = ["weight_kg", "height_cm", "age", "phase_encoded"]
    X = df[features]
    y = df["bmr_measured"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("=" * 50)
    print("BMR MODEL — LEARNED COEFFICIENTS")
    print("=" * 50)
    print(f"  weight_kg    : {model.coef_[0]:.4f}  (Harris-Benedict: 9.563)")
    print(f"  height_cm    : {model.coef_[1]:.4f}  (Harris-Benedict: 1.850)")
    print(f"  age          : {model.coef_[2]:.4f}  (Harris-Benedict: -4.676)")
    print(f"  phase_encoded: {model.coef_[3]:.4f}  (Harris-Benedict: not present)")
    print(f"  intercept    : {model.intercept_:.4f}  (Harris-Benedict: 655.1)")
    print(f"\n  MAE : {mae:.2f} kcal")
    print(f"  R²  : {r2:.4f}")

    os.makedirs("ml/models", exist_ok=True)
    with open("ml/models/bmr_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model


# ── VO2 Model ────────────────────────────────────────────────
def train_vo2_model(df: pd.DataFrame):
    """
    Learns female-specific VO2 coefficients with interaction terms.
    Features: hr_ratio, hemoglobin, fitness_tier, phase + interactions
    Target: vo2_measured
    """
    df = encode_phase(df)

    # Base feature
    df["true_max_hr"] = 206 - (0.88 * df["age"])
    df["hr_ratio"]    = df["true_max_hr"] / df["resting_hr"]

    # Fitness tier from AHA rule
    df["fitness_tier"] = df["resting_hr"].apply(
        lambda hr: 3 if hr < 60 else (2 if hr <= 75 else 1)
    )

    # Interaction terms — this is what captures compounding effects
    df["hb_x_phase"]     = df["hemoglobin_g_dl"] * df["phase_encoded"]
    df["hr_x_fitness"]   = df["hr_ratio"] * df["fitness_tier"]
    df["hb_x_fitness"]   = df["hemoglobin_g_dl"] * df["fitness_tier"]

    features = [
        "hr_ratio",
        "hemoglobin_g_dl",
        "fitness_tier",
        "phase_encoded",
        "hb_x_phase",       # hemoglobin × phase interaction
        "hr_x_fitness",     # hr ratio × fitness interaction
        "hb_x_fitness"      # hemoglobin × fitness interaction
    ]

    X = df[features]
    y = df["vo2_measured"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("VO2 MODEL — LEARNED COEFFICIENTS")
    print("=" * 50)
    for fname, coef in zip(features, model.coef_):
        print(f"  {fname:<20}: {coef:.4f}")
    print(f"  intercept           : {model.intercept_:.4f}")
    print(f"  (Uth-Sorensen used fixed coefficient: 15.0)")
    print(f"\n  MAE : {mae:.2f} ml/kg/min")
    print(f"  R²  : {r2:.4f}")

    with open("ml/models/vo2_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, df


# ── Quantile Regression (Confidence Intervals) ──────────────
def train_confidence_models(df: pd.DataFrame):
    """
    Trains lower (10th) and upper (90th) quantile models.
    Gives confidence interval instead of single point estimate.
    """
    df = encode_phase(df)
    df["true_max_hr"] = 206 - (0.88 * df["age"])
    df["hr_ratio"]    = df["true_max_hr"] / df["resting_hr"]
    df["fitness_tier"] = df["resting_hr"].apply(
        lambda hr: 3 if hr < 60 else (2 if hr <= 75 else 1)
    )
    df["hb_x_phase"]   = df["hemoglobin_g_dl"] * df["phase_encoded"]
    df["hr_x_fitness"] = df["hr_ratio"] * df["fitness_tier"]
    df["hb_x_fitness"] = df["hemoglobin_g_dl"] * df["fitness_tier"]

    features = [
        "hr_ratio", "hemoglobin_g_dl", "fitness_tier",
        "phase_encoded", "hb_x_phase", "hr_x_fitness", "hb_x_fitness"
    ]

    X = df[features]
    y = df["vo2_measured"]

    lower_model = QuantileRegressor(quantile=0.10, alpha=0, solver="highs")
    upper_model = QuantileRegressor(quantile=0.90, alpha=0, solver="highs")

    lower_model.fit(X, y)
    upper_model.fit(X, y)

    print("\n" + "=" * 50)
    print("CONFIDENCE INTERVAL MODELS (10th / 90th percentile)")
    print("=" * 50)
    print("  Lower bound model trained ✅")
    print("  Upper bound model trained ✅")

    with open("ml/models/vo2_lower.pkl", "wb") as f:
        pickle.dump(lower_model, f)
    with open("ml/models/vo2_upper.pkl", "wb") as f:
        pickle.dump(upper_model, f)

    return lower_model, upper_model


# ── Run All ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/femfit_nhanes_women.csv")
    print(f"Dataset: {len(df)} rows\n")

    bmr_model              = train_bmr_model(df)
    vo2_model, df_enriched = train_vo2_model(df)
    lower, upper           = train_confidence_models(df_enriched)

    print("\n" + "=" * 50)
    print("All models saved to ml/models/")
    print("=" * 50)