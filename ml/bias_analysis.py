import pandas as pd
import numpy as np
from ml.femfit_engine import run_on_dataset
from scipy.stats import pearsonr

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv("data/femfit_nhanes_women.csv")
print(f"Dataset loaded: {len(df)} rows\n")

# ── Run Engine ───────────────────────────────────────────────
results = run_on_dataset(df)

# ── Benchmark Comparison ─────────────────────────────────────
# How close is standard vs femfit to the measured benchmark?
results["bmr_error_standard"] = (results["bmr_standard"] - results["bmr_measured"]).abs()
results["bmr_error_femfit"]   = (results["bmr_femfit"]   - results["bmr_measured"]).abs()
results["vo2_error_standard"] = (results["vo2_standard"] - results["vo2_measured"]).abs()
results["vo2_error_femfit"]   = (results["vo2_femfit"]   - results["vo2_measured"]).abs()

results["bmr_improvement"] = results["bmr_error_standard"] - results["bmr_error_femfit"]
results["vo2_improvement"] = results["vo2_error_standard"] - results["vo2_error_femfit"]

# ── Report 1 — Overall Benchmark Comparison ──────────────────
print("=" * 60)
print("REPORT 1 — BENCHMARK COMPARISON (Standard vs FemFit)")
print("=" * 60)

bmr_std_mae  = results["bmr_error_standard"].mean()
bmr_fem_mae  = results["bmr_error_femfit"].mean()
bmr_improve  = ((bmr_std_mae - bmr_fem_mae) / bmr_std_mae * 100)

vo2_std_mae  = results["vo2_error_standard"].mean()
vo2_fem_mae  = results["vo2_error_femfit"].mean()
vo2_improve  = ((vo2_std_mae - vo2_fem_mae) / vo2_std_mae * 100)

print(f"\nBMR")
print(f"  Standard MAE vs measured : {bmr_std_mae:.2f} kcal")
print(f"  FemFit   MAE vs measured : {bmr_fem_mae:.2f} kcal")
print(f"  Error reduction          : {bmr_improve:.1f}%")

print(f"\nVO2 Max")
print(f"  Standard MAE vs measured : {vo2_std_mae:.2f} ml/kg/min")
print(f"  FemFit   MAE vs measured : {vo2_fem_mae:.2f} ml/kg/min")
print(f"  Error reduction          : {vo2_improve:.1f}%")

# ── Report 2 — Phase-wise Benchmark ──────────────────────────
print("\n" + "=" * 60)
print("REPORT 2 — ERROR BY CYCLE PHASE")
print("=" * 60)

phase_report = results.groupby("cycle_phase").agg(
    count                = ("age", "count"),
    bmr_std_mae          = ("bmr_error_standard", "mean"),
    bmr_fem_mae          = ("bmr_error_femfit",   "mean"),
    vo2_std_mae          = ("vo2_error_standard", "mean"),
    vo2_fem_mae          = ("vo2_error_femfit",   "mean"),
).round(2)

phase_report["bmr_improvement_%"] = (
    (phase_report["bmr_std_mae"] - phase_report["bmr_fem_mae"])
    / phase_report["bmr_std_mae"] * 100
).round(1)

phase_report["vo2_improvement_%"] = (
    (phase_report["vo2_std_mae"] - phase_report["vo2_fem_mae"])
    / phase_report["vo2_std_mae"] * 100
).round(1)

print(phase_report.to_string())

# ── Report 3 — Fitness Tier Breakdown ────────────────────────
print("\n" + "=" * 60)
print("REPORT 3 — ERROR BY FITNESS TIER")
print("=" * 60)

results["fitness_tier"] = results["resting_hr"].apply(
    lambda hr: "high" if hr < 60 else ("moderate" if hr <= 75 else "low")
)

fitness_report = results.groupby("fitness_tier").agg(
    count       = ("age", "count"),
    bmr_std_mae = ("bmr_error_standard", "mean"),
    bmr_fem_mae = ("bmr_error_femfit",   "mean"),
    vo2_std_mae = ("vo2_error_standard", "mean"),
    vo2_fem_mae = ("vo2_error_femfit",   "mean"),
).round(2)

fitness_report["bmr_improvement_%"] = (
    (fitness_report["bmr_std_mae"] - fitness_report["bmr_fem_mae"])
    / fitness_report["bmr_std_mae"] * 100
).round(1)

fitness_report["vo2_improvement_%"] = (
    (fitness_report["vo2_std_mae"] - fitness_report["vo2_fem_mae"])
    / fitness_report["vo2_std_mae"] * 100
).round(1)

print(fitness_report.to_string())

# ── Report 4 — Most Impacted Users ───────────────────────────
print("\n" + "=" * 60)
print("REPORT 4 — REAL WORLD IMPACT (Most Affected Women)")
print("=" * 60)

worst = results.nlargest(5, "bmr_improvement")[
    ["age", "weight_kg", "cycle_phase", "fitness_tier",
     "bmr_standard", "bmr_femfit", "bmr_measured",
     "bmr_error_standard", "bmr_error_femfit", "bmr_improvement"]
].round(2)
print("\nTop 5 women most helped by FemFit (BMR):")
print(worst.to_string(index=False))

# ── Report 5 — Significance ───────────────────────────────────
print("\n" + "=" * 60)
print("REPORT 5 — SIGNIFICANCE THRESHOLDS")
print("=" * 60)

bmr_sig_std = (results["bmr_error_standard"] >= 50).sum()
bmr_sig_fem = (results["bmr_error_femfit"]   >= 50).sum()
vo2_sig_std = (results["vo2_error_standard"] >= 5).sum()
vo2_sig_fem = (results["vo2_error_femfit"]   >= 5).sum()

print(f"\n  BMR error >= 50 kcal")
print(f"    Standard : {bmr_sig_std} / {len(results)} ({100*bmr_sig_std/len(results):.1f}%)")
print(f"    FemFit   : {bmr_sig_fem} / {len(results)} ({100*bmr_sig_fem/len(results):.1f}%)")

print(f"\n  VO2 error >= 5 ml/kg/min")
print(f"    Standard : {vo2_sig_std} / {len(results)} ({100*vo2_sig_std/len(results):.1f}%)")
print(f"    FemFit   : {vo2_sig_fem} / {len(results)} ({100*vo2_sig_fem/len(results):.1f}%)")

# ── Report 6 — VO2 Proxy Validation (HRR Correlation) ────────
print("\n" + "=" * 60)
print("REPORT 6 — VO2 VALIDATION VIA HEART RATE RECOVERY")
print("=" * 60)


corr_standard, p_standard = pearsonr(results["vo2_standard"], results["hr_recovery"])
corr_femfit,   p_femfit   = pearsonr(results["vo2_femfit"],   results["hr_recovery"])

print(f"\n  Standard VO2 correlation with HRR : r = {corr_standard:.4f}  (p = {p_standard:.4f})")
print(f"  FemFit   VO2 correlation with HRR : r = {corr_femfit:.4f}  (p = {p_femfit:.4f})")
print(f"\n  Improvement in correlation        : {round(corr_femfit - corr_standard, 4)}")

if corr_femfit > corr_standard:
    print(f"\n  ✅ FemFit VO2 predicts real cardiovascular")
    print(f"     performance better than standard formula")
else:
    print(f"\n  Standard correlation higher — investigate further")

# Phase-wise correlation
print(f"\n  Phase-wise HRR correlation:")
for phase in ["menstrual", "follicular", "ovulatory", "luteal"]:
    subset = results[results["cycle_phase"] == phase]
    r_std, _ = pearsonr(subset["vo2_standard"], subset["hr_recovery"])
    r_fem, _ = pearsonr(subset["vo2_femfit"],   subset["hr_recovery"])
    print(f"    {phase:<12} standard: {r_std:.4f}  femfit: {r_fem:.4f}  delta: {round(r_fem-r_std, 4)}")
# ── Report 7 — Calorie Burn Bias ─────────────────────────────
print("\n" + "=" * 60)
print("REPORT 7 — CALORIE BURN BIAS (Standard vs FemFit)")
print("=" * 60)

from ml.calorie_model import predict_calories

# Generate FemFit calorie predictions for all 1263 rows
print("Running calorie predictions on 1263 rows...")
calorie_femfit = []
for _, row in results.iterrows():
    pred = predict_calories(
        row["weight_kg"], int(row["age"]),
        row["resting_hr"], row["hemoglobin_g_dl"],
        row["cycle_phase"]
    )
    calorie_femfit.append(pred["calories_femfit"])

results["calorie_femfit"]    = calorie_femfit
results["calorie_standard"]  = 5.0 * results["weight_kg"].clip(upper=120) * 0.5
results["calorie_error_std"] = (results["calorie_standard"] - results["calories_measured"]).abs()
results["calorie_error_fem"] = (results["calorie_femfit"]   - results["calories_measured"]).abs()
results["calorie_improvement"] = results["calorie_error_std"] - results["calorie_error_fem"]

cal_std_mae = results["calorie_error_std"].mean()
cal_fem_mae = results["calorie_error_fem"].mean()
cal_improve = (cal_std_mae - cal_fem_mae) / cal_std_mae * 100

print(f"\n  Standard MAE vs measured : {cal_std_mae:.2f} kcal")
print(f"  FemFit   MAE vs measured : {cal_fem_mae:.2f} kcal")
print(f"  Error reduction          : {cal_improve:.1f}%")

# Phase breakdown
print(f"\n  Phase-wise calorie error:")
cal_phase = results.groupby("cycle_phase").agg(
    count           = ("age", "count"),
    std_mae         = ("calorie_error_std", "mean"),
    fem_mae         = ("calorie_error_fem", "mean"),
).round(2)
cal_phase["improvement_%"] = (
    (cal_phase["std_mae"] - cal_phase["fem_mae"])
    / cal_phase["std_mae"] * 100
).round(1)
print(cal_phase.to_string())

# Significance
cal_sig_std = (results["calorie_error_std"] >= 20).sum()
cal_sig_fem = (results["calorie_error_fem"] >= 20).sum()
print(f"\n  Calorie error >= 20 kcal")
print(f"    Standard : {cal_sig_std} / {len(results)} ({100*cal_sig_std/len(results):.1f}%)")
print(f"    FemFit   : {cal_sig_fem} / {len(results)} ({100*cal_sig_fem/len(results):.1f}%)")

# ── Report 8 — HR Zone Bias ───────────────────────────────────
print("\n" + "=" * 60)
print("REPORT 8 — HR ZONE BIAS (Standard vs FemFit)")
print("=" * 60)

from ml.femfit_engine import hr_zones

def get_zone_label(hr: float, max_hr: float) -> str:
    pct = hr / max_hr
    if pct < 0.60:   return "zone_1_rest"
    elif pct < 0.70: return "zone_2_fat_burn"
    elif pct < 0.80: return "zone_3_cardio"
    elif pct < 0.90: return "zone_4_peak"
    else:            return "zone_5_max"

# Simulate exercise HR at 75% effort for each user
results["exercise_hr"] = (results["resting_hr"] + 
    (results["resting_hr"] * 0.75)).clip(upper=185).round(0)

# Zone classification
std_zones  = []
fem_zones  = []
mismatches = 0

for _, row in results.iterrows():
    std_max = 220 - row["age"]
    fem_max = (206 - 0.88 * row["age"]) * {
        "menstrual": 0.97, "follicular": 1.00,
        "ovulatory": 1.00, "luteal": 0.96
    }[row["cycle_phase"]]

    std_zone = get_zone_label(row["exercise_hr"], std_max)
    fem_zone = get_zone_label(row["exercise_hr"], fem_max)

    std_zones.append(std_zone)
    fem_zones.append(fem_zone)
    if std_zone != fem_zone:
        mismatches += 1

results["std_zone"] = std_zones
results["fem_zone"] = fem_zones

mismatch_pct = 100 * mismatches / len(results)
print(f"\n  Women in wrong HR zone (standard): {mismatches} / {len(results)} ({mismatch_pct:.1f}%)")

# Phase-wise mismatch
results["zone_mismatch"] = results["std_zone"] != results["fem_zone"]
phase_mismatch = results.groupby("cycle_phase")["zone_mismatch"].agg(
    ["sum", "count"]
).round(2)
phase_mismatch["mismatch_%"] = (
    phase_mismatch["sum"] / phase_mismatch["count"] * 100
).round(1)
phase_mismatch.columns = ["mismatches", "total", "mismatch_%"]
print(f"\n  Phase-wise HR zone mismatch:")
print(phase_mismatch.to_string())

# Zone confusion — what zone are they really in?
print(f"\n  Most common zone misclassifications:")
mismatch_df = results[results["zone_mismatch"]][["std_zone", "fem_zone", "cycle_phase"]]
confusion = mismatch_df.groupby(
    ["std_zone", "fem_zone"]
).size().sort_values(ascending=False).head(5)
print(confusion.to_string())

# ── Save ──────────────────────────────────────────────────────
results.to_csv("data/bias_report.csv", index=False)
print("\nFull results saved to data/bias_report.csv")