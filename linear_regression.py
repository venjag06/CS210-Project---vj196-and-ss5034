import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

DB_PATH    = "housing.db"
OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# STEP 1: Load merged table
conn   = sqlite3.connect(DB_PATH)
df     = pd.read_sql("SELECT * FROM merged", conn)
conn.close()
print(f"Loaded {len(df):,} rows")


# Preparing features
#
# Features we'll use:
#   - sqfeet, beds, baths       (property size)
#   - type                      (one hot encoded tho)
#   - median_income             (city wealth)
#   - unemployment_rate         (city economic health)
#   - bachelors_rate            (city education level)
#   - median_rent_census        (Census rent benchmark)
#
# Target: log(price) — log-transform makes the
# skewed price distribution more normal, which
# helps linear regression work better.
# ─────────────────────────────────────────────

FEATURES = [
    "sqfeet", "beds", "baths",
    "median_income", "unemployment_rate",
    "bachelors_rate", "median_rent_census",
]

# One-hot encode the 'type' column (apartment, house, etc.)
type_dummies = pd.get_dummies(df["type"], prefix="type", drop_first=True)

# Build feature matrix X
X = pd.concat([df[FEATURES], type_dummies], axis=1)
X = X.fillna(0)  # fill any remaining nulls

# Log-transform target
y = np.log(df["price"])

print(f"Features: {X.columns.tolist()}")
print(f"X shape:  {X.shape}")


# Train / Test split (80 / 20) psuedorandom (random_state) for replicatability 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")


# Scale features
#
# Linear Regression is sensitive to scale —
# sqfeet is in the thousands while unemployment_rate
# is in single digits, so we standardize everything
# to have mean=0 and std=1.

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# Fit Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("\nModel trained")


# Mode evaluation on test set
#
# We predict log(price), then exponentiate back
# to get the actual dollar prediction.
# RMSE is in dollars tells us the average error of our price predictions.
# R-squared (0–1) tells us how much of the price variation the model explains. Closer to 1 is better.

log_preds  = model.predict(X_test_scaled)
preds      = np.exp(log_preds)          # convert back to dollars
actuals    = np.exp(y_test)             # convert back to dollars

rmse = np.sqrt(mean_squared_error(actuals, preds))
r2   = r2_score(y_test, log_preds)     # R^2 on log scale

print(f"\n── Linear Regression Results ──────────────")
print(f"  RMSE:  ${rmse:,.2f}  (avg prediction error in dollars)")
print(f"  R²:    {r2:.4f}    (1.0 = perfect, 0.0 = no better than mean)")


# ─────────────────────────────────────────────
# Feature importances (coefficients)
# After scaling, larger absolute coefficient = bigger impact on predicted price.

coef_df = pd.DataFrame({
    "feature":     X.columns,
    "coefficient": model.coef_,
}).sort_values("coefficient", key=abs, ascending=False)

print("\n── Feature Coefficients (sorted by impact) ──")
print(coef_df.to_string(index=False))


# Plot — Actual vs Predicted prices
sns.set_theme(style="whitegrid")

sample_idx = np.random.choice(len(actuals), size=2000, replace=False)
act_sample  = actuals.values[sample_idx]
pred_sample = preds[sample_idx]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(act_sample, pred_sample, alpha=0.2, s=10, color="#004BC3")
max_val = max(act_sample.max(), pred_sample.max())
ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_title("Linear Regression: Actual vs Predicted Rent", fontsize=13, fontweight="bold")
ax.set_xlabel("Actual Rent ($)")
ax.set_ylabel("Predicted Rent ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_lr_actual_vs_predicted.png", dpi=150)
plt.close()
print(f"\n✓ Plot saved: {OUTPUT_DIR}/07_lr_actual_vs_predicted.png")


# Save predictions for ALL rows to DB
# We predict on the full dataset (not just test) so we can later compare every listing's predicted price vs its actual listed price.
X_all_scaled   = scaler.transform(X)
log_preds_all  = model.predict(X_all_scaled)
predicted_all  = np.exp(log_preds_all)

results_df = df[["id", "region", "state", "price", "type",
                  "sqfeet", "beds", "baths"]].copy()
results_df["predicted_price_lr"] = predicted_all.round(2)
results_df["price_diff_lr"]      = (df["price"] - predicted_all).round(2)
results_df["pct_diff_lr"]        = ((df["price"] - predicted_all) / predicted_all * 100).round(2)

conn = sqlite3.connect(DB_PATH)
results_df.to_sql("predictions_lr", conn, if_exists="replace", index=False)
conn.close()
print(f"saved {len(results_df):,} predictions to 'predictions_lr' table")

print("\n── Sample predictions ──────────────────────")
print(results_df[["region", "state", "price", "predicted_price_lr",
                   "pct_diff_lr"]].head(10).to_string(index=False))
print("\nDone")