import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

PHASE_ENCODING = {
    "menstrual":  0,
    "follicular": 1,
    "ovulatory":  2,
    "luteal":     3
}


def train_calorie_model(df: pd.DataFrame):
    df = df.copy()

    # Encode phase
    df["phase_encoded"] = df["cycle_phase"].map(PHASE_ENCODING)

    # Fitness tier
    df["fitness_tier"] = df["resting_hr"].apply(
        lambda hr: 3 if hr < 60 else (2 if hr <= 75 else 1)
    )

    # Interaction terms
    df["phase_x_fitness"]  = df["phase_encoded"] * df["fitness_tier"]
    df["hb_x_phase"]       = df["hemoglobin_g_dl"] * df["phase_encoded"]
    df["weight_x_fitness"] = df["weight_kg"].clip(upper=120) * df["fitness_tier"]

    features = [
        "weight_kg",
        "age",
        "resting_hr",
        "hemoglobin_g_dl",
        "phase_encoded",
        "fitness_tier",
        "phase_x_fitness",
        "hb_x_phase",
        "weight_x_fitness"
    ]

    X = df[features]
    y = df["calories_measured"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("=" * 55)
    print("CALORIE MODEL — XGBoost")
    print("=" * 55)
    print(f"  MAE : {mae:.2f} kcal")
    print(f"  R²  : {r2:.4f}")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print(f"\n  Feature Importance:")
    for feat, imp in importance.items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<20} {imp:.4f}  {bar}")

    os.makedirs("ml/models", exist_ok=True)
    with open("ml/models/calorie_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("ml/models/calorie_features.pkl", "wb") as f:
        pickle.dump(features, f)

    return model, features

def explain_prediction(weight_kg: float, age: int, resting_hr: float,
                       hemoglobin: float, cycle_phase: str) -> dict:
    """
    SHAP explanation for a single prediction.
    Shows exactly WHY the model gave this calorie estimate.
    """
    import shap

    with open("ml/models/calorie_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ml/models/calorie_features.pkl", "rb") as f:
        features = pickle.load(f)

    phase_enc     = PHASE_ENCODING[cycle_phase]
    fitness_tier  = 3 if resting_hr < 60 else (2 if resting_hr <= 75 else 1)
    weight_capped = min(weight_kg, 120)

    X = pd.DataFrame([[
        weight_kg, age, resting_hr, hemoglobin,
        phase_enc, fitness_tier,
        phase_enc * fitness_tier,
        hemoglobin * phase_enc,
        weight_capped * fitness_tier
    ]], columns=features)

    # SHAP explainer
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Build explanation dict
    contributions = {}
    for feat, shap_val in zip(features, shap_values[0]):
        contributions[feat] = round(float(shap_val), 3)

    # Sort by absolute impact
    sorted_contributions = dict(
        sorted(contributions.items(),
               key=lambda x: abs(x[1]),
               reverse=True)
    )

    # Human readable explanation
    explanation_lines = []
    for feat, val in sorted_contributions.items():
        direction = "+" if val > 0 else ""
        explanation_lines.append(f"{feat}: {direction}{val} kcal")

    return {
        "prediction":    round(float(model.predict(X)[0]), 2),
        "contributions": sorted_contributions,
        "explanation":   explanation_lines
    }


def predict_calories(weight_kg: float, age: int, resting_hr: float,
                     hemoglobin: float, cycle_phase: str) -> dict:
    with open("ml/models/calorie_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ml/models/calorie_features.pkl", "rb") as f:
        features = pickle.load(f)

    phase_enc    = PHASE_ENCODING[cycle_phase]
    fitness_tier = 3 if resting_hr < 60 else (2 if resting_hr <= 75 else 1)
    weight_capped = min(weight_kg, 120)

    X = pd.DataFrame([[
        weight_kg, age, resting_hr, hemoglobin,
        phase_enc, fitness_tier,
        phase_enc * fitness_tier,
        hemoglobin * phase_enc,
        weight_capped * fitness_tier
    ]], columns=features)

    predicted = round(float(model.predict(X)[0]), 2)

    # Standard formula (what wearables use)
    standard = round(5.0 * min(weight_kg, 120) * 0.5, 2)

    return {
        "calories_standard": standard,
        "calories_femfit":   predicted,
        "calories_delta":    round(predicted - standard, 2),
        "calories_bias_pct": round((predicted - standard) / standard * 100, 2)
    }


if __name__ == "__main__":
    df = pd.read_csv("data/femfit_nhanes_women.csv")
    print(f"Dataset: {len(df)} rows\n")

    model, features = train_calorie_model(df)

    # Test predictions across phases
    print(f"\n  Phase comparison (same woman, different phases):")
    print(f"  {'Phase':<12} {'Standard':>10} {'FemFit':>10} {'Delta':>8}")
    print(f"  {'-'*45}")

    for phase in ["menstrual", "follicular", "ovulatory", "luteal"]:
        result = predict_calories(62.1, 25, 86, 12.5, phase)
        print(f"  {phase:<12} "
              f"{result['calories_standard']:>10} "
              f"{result['calories_femfit']:>10} "
              f"{result['calories_delta']:>8}")
    
    # SHAP explanation
    print(f"\n{'='*55}")
    print("SHAP EXPLANATION — Why this prediction?")
    print(f"{'='*55}")
    explanation = explain_prediction(62.1, 25, 86, 12.5, "luteal")
    print(f"  Predicted: {explanation['prediction']} kcal\n")
    print(f"  Feature contributions:")
    for line in explanation["explanation"]:
        print(f"    {line}")

    # Compare same woman across phases
    print(f"\n  Phase-wise SHAP — phase contribution only:")
    for phase in ["menstrual", "follicular", "ovulatory", "luteal"]:
        exp = explain_prediction(62.1, 25, 86, 12.5, phase)
        phase_contribution = exp["contributions"].get("phase_encoded", 0)
        print(f"    {phase:<12} phase impact: {phase_contribution:+.3f} kcal")