import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

DB_PATH    = "housing.db"
OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load merged table
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM merged", conn)
conn.close()
print(f"Loaded {len(df):,} rows")


# Prepare features (same as Linear regression

FEATURES = [
    "sqfeet", "beds", "baths",
    "median_income", "unemployment_rate",
    "bachelors_rate", "median_rent_census",
]

type_dummies = pd.get_dummies(df["type"], prefix="type", drop_first=True)
X = pd.concat([df[FEATURES], type_dummies], axis=1).fillna(0)
y = np.log(df["price"])  # log-transform target

print(f"Features: {X.columns.tolist()}")
print(f"X shape:  {X.shape}")


# Train / Test split (80 / 20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")


# Scale features
# Scaling is importnt for KNN cus it uses distance to find neighbors. Without scaling, sqfeet
# (in the thousands) would completely overpower unemployment_rate (in single digits), meaning
# the "nearest neighbors" would just be listings with similar square footage and ignore everything else.

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


#  Finding the best k using cross-validation
# We try k = 3, 5, 7, ..., 25 and pick the k right before the root mean squared error begins to diverge 
# aka the k with lowest root mean squared error

print("\nFinding best k via cross-validation")

k_values = range(3, 26, 2)  # 3, 5, 7, ..., 25
cv_rmses = []

# Sample to speed up CV — 20k rows is enough to find a good k
sample_size = min(20_000, len(X_train))
idx = np.random.choice(len(X_train), size=sample_size, replace=False)
X_cv = X_train_scaled[idx]
y_cv = y_train.values[idx]

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    scores = cross_val_score(knn, X_cv, y_cv,
                             cv=5, scoring="neg_root_mean_squared_error")
    cv_rmses.append(-scores.mean())
    print(f"  k={k:2d}  CV RMSE (log scale): {-scores.mean():.4f}")

best_k = list(k_values)[np.argmin(cv_rmses)]
#the k with lowest root mean squared error is the k we'll use. 
print(f"\nBest k = {best_k}")

#to get the best k another thing we can do is compare each k RMSE to the previous k RMSE  (the way we learned in class)
# and find the k right before the RMSE starts to diverge (increase significantly).
# however we just try all k's up to 25 and pick the one with the lowest RMSE, which is simpler and more straightforward.


# plotting CV RMSE vs k
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_values), cv_rmses, marker="o", color="#004BC3", linewidth=2)
ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5,
           label=f"Best k = {best_k}")
ax.set_title("KNN Cross-Validation: RMSE vs k", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Neighbors (k)")
ax.set_ylabel("CV RMSE (log scale)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_knn_cv_rmse.png", dpi=150)
plt.close()
print(f"✓ Plot saved: {OUTPUT_DIR}/08_knn_cv_rmse.png")


# Fit final KNN model on full training set
# weights="distance" means closer neighbors have more influence than farther ones 
#  almost always better than treating all k neighbors equally.
# ─────────────────────────────────────────────
print(f"\nFitting KNN (k={best_k}) on full training set...")
knn_model = KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)
print("KNN Model trained")


# model evaluation on test set 

log_preds = knn_model.predict(X_test_scaled)
preds     = np.exp(log_preds)
actuals   = np.exp(y_test)

rmse = np.sqrt(mean_squared_error(actuals, preds))
r2   = r2_score(y_test, log_preds)

print(f"\n── KNN Regression Results (k={best_k}) ────────")
print(f"  RMSE:  ${rmse:,.2f}  (avg prediction error in dollars)")
print(f"  R^2:    {r2:.4f}    (1.0 = perfect, 0.0 = no better than mean)")


# Plotting Actual vs Predicted
sample_idx  = np.random.choice(len(actuals), size=2000, replace=False)
act_sample  = actuals.values[sample_idx]
pred_sample = preds[sample_idx]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(act_sample, pred_sample, alpha=0.2, s=10, color="#004BC3")
max_val = max(act_sample.max(), pred_sample.max())
ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_title(f"KNN (k={best_k}): Actual vs Predicted Rent", fontsize=13, fontweight="bold")
ax.set_xlabel("Actual Rent ($)")
ax.set_ylabel("Predicted Rent ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_knn_actual_vs_predicted.png", dpi=150)
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/09_knn_actual_vs_predicted.png")


# now we Save predictions for TEST rows only
# We only predict on rows the model never saw
# during training. This avoids KNN memorizing
# its own training points (distance = 0 → price echoed back exactly), giving honest predictions.

predicted_test = np.exp(log_preds)   # already computed in Step 7

# X_test preserves the original DataFrame index
test_rows = df.loc[X_test.index, ["id", "region", "state", "price",
                                   "type", "sqfeet", "beds", "baths"]].copy()
test_rows["predicted_price_knn"] = predicted_test.round(2)
test_rows["price_diff_knn"]      = (test_rows["price"] - predicted_test).round(2)
test_rows["pct_diff_knn"]        = ((test_rows["price"] - predicted_test) / predicted_test * 100).round(2)
test_rows.reset_index(drop=True, inplace=True)

conn = sqlite3.connect(DB_PATH)
test_rows.to_sql("predictions_knn", conn, if_exists="replace", index=False)
conn.close()
print(f"Saved {len(test_rows):,} test-set predictions to 'predictions_knn' table")

print("\n── Sample predictions (test set only) ─────")
print(test_rows[["region", "state", "price", "predicted_price_knn",
                  "pct_diff_knn"]].head(10).to_string(index=False))
print("\nDone")

# for prediction
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(knn_model, "models/knn_model.joblib")
joblib.dump(scaler,    "models/knn_scaler.joblib")
print("\n✓ Saved: models/knn_model.joblib, models/knn_scaler.joblib")
print("\nDone!")