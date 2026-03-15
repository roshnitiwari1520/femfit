import pandas as pd
import numpy as np
from ml.femfit_engine import run_on_dataset

# --- Load Dataset ---
df = pd.read_csv("data/femfit_nhanes_women.csv")
print(f"Dataset loaded: {len(df)} rows\n")

# --- Run Engine ---
results = run_on_dataset(df)

# --- Compute Deltas ---
results["bmr_delta"] = (results["bmr_corrected"] - results["bmr_standard"]).round(2)
results["vo2_delta"] = (results["vo2_corrected"] - results["vo2_standard"]).round(2)

# -----------------------------------------------
# REPORT 1 — Overall Bias Summary
# -----------------------------------------------
print("=" * 50)
print("OVERALL BIAS SUMMARY")
print("=" * 50)

print(f"\nBMR")
print(f"  Mean standard  : {results['bmr_standard'].mean():.2f} kcal")
print(f"  Mean corrected : {results['bmr_corrected'].mean():.2f} kcal")
print(f"  Mean delta     : {results['bmr_delta'].mean():.2f} kcal")
print(f"  Max overestimate: {results['bmr_delta'].min():.2f} kcal")
print(f"  Max underestimate: {results['bmr_delta'].max():.2f} kcal")

print(f"\nVO2 Max")
print(f"  Mean standard  : {results['vo2_standard'].mean():.2f} ml/kg/min")
print(f"  Mean corrected : {results['vo2_corrected'].mean():.2f} ml/kg/min")
print(f"  Mean delta     : {results['vo2_delta'].mean():.2f} ml/kg/min")
print(f"  Max overestimate: {results['vo2_delta'].min():.2f} ml/kg/min")
print(f"  Max underestimate: {results['vo2_delta'].max():.2f} ml/kg/min")

# -----------------------------------------------
# REPORT 2 — Phase-wise Breakdown
# -----------------------------------------------
print("\n" + "=" * 50)
print("PHASE-WISE BREAKDOWN")
print("=" * 50)

phase_report = results.groupby("cycle_phase").agg(
    count           = ("age", "count"),
    bmr_std_mean    = ("bmr_standard", "mean"),
    bmr_corr_mean   = ("bmr_corrected", "mean"),
    bmr_delta_mean  = ("bmr_delta", "mean"),
    vo2_std_mean    = ("vo2_standard", "mean"),
    vo2_corr_mean   = ("vo2_corrected", "mean"),
    vo2_delta_mean  = ("vo2_delta", "mean"),
).round(2)

print(phase_report.to_string())

# -----------------------------------------------
# REPORT 3 — Worst Affected Users
# -----------------------------------------------
print("\n" + "=" * 50)
print("TOP 5 MOST BIASED BMR CASES")
print("=" * 50)
worst_bmr = results.nsmallest(5, "bmr_delta")[
    ["age", "weight_kg", "height_cm", "cycle_phase",
     "bmr_standard", "bmr_corrected", "bmr_delta"]
]
print(worst_bmr.to_string(index=False))

print("\n" + "=" * 50)
print("TOP 5 MOST BIASED VO2 CASES")
print("=" * 50)
worst_vo2 = results.nsmallest(5, "vo2_delta")[
    ["age", "resting_hr", "hemoglobin_g_dl", "cycle_phase",
     "vo2_standard", "vo2_corrected", "vo2_delta"]
]
print(worst_vo2.to_string(index=False))

# -----------------------------------------------
# REPORT 4 — % of Users Significantly Affected
# -----------------------------------------------
print("\n" + "=" * 50)
print("SIGNIFICANCE THRESHOLDS")
print("=" * 50)

bmr_sig  = (results["bmr_delta"].abs() >= 50).sum()
vo2_sig  = (results["vo2_delta"].abs() >= 5).sum()

print(f"  Users with BMR error >= 50 kcal  : {bmr_sig} / {len(results)} "
      f"({100*bmr_sig/len(results):.1f}%)")
print(f"  Users with VO2 error >= 5 ml/kg  : {vo2_sig} / {len(results)} "
      f"({100*vo2_sig/len(results):.1f}%)")

# --- Save full results ---
results.to_csv("data/bias_report.csv", index=False)
print("\nFull results saved to data/bias_report.csv")