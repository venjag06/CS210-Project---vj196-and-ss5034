import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

DB_PATH    = "housing.db"
OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
#creating directory to put our visualizations

#loading merged table
conn   = sqlite3.connect(DB_PATH)
merged = pd.read_sql("SELECT * FROM merged", conn)
conn.close()
print(f"Loaded {len(merged):,} rows\n")

sns.set_theme(style="whitegrid", palette="muted")


# PLOT 1: Distribution of rental prices
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(merged["price"], bins=80, color="#004BC3", edgecolor="white", linewidth=0.4)
ax.set_title("Distribution of Rental Prices", fontsize=14, fontweight="bold")
ax.set_xlabel("Monthly Rent ($)")
ax.set_ylabel("Number of Listings")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
median_price = merged["price"].median()
ax.axvline(median_price, color="crimson", linestyle="--", linewidth=1.5,
           label=f"Median: ${median_price:,.0f}")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_price_distribution.png", dpi=150)
plt.close()
print("Plot 1: Price distribution saved")


# PLOT 2: Price vs Square Footage (scatter)
# Sample 5,000 points so the plot isn't overwhelming
sample = merged.sample(n=5_000, random_state=42)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(sample["sqfeet"], sample["price"],
           alpha=0.25, s=10, color="#004BC3")
ax.set_title("Rental Price vs. Square Footage", fontsize=14, fontweight="bold")
ax.set_xlabel("Square Footage")
ax.set_ylabel("Monthly Rent ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_price_vs_sqfeet.png", dpi=150)
plt.close()
print("Plot 2: Price vs sqfeet saved")


# PLOT 3: Average rent by state (bar chart)
avg_by_state = (
    merged.groupby("state")["price"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
avg_by_state.columns = ["state", "avg_price"]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(avg_by_state["state"], avg_by_state["avg_price"],
              color="#004BC3", edgecolor="white", linewidth=0.4)
ax.set_title("Average Monthly Rent by State", fontsize=14, fontweight="bold")
ax.set_xlabel("State")
ax.set_ylabel("Average Rent ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_avg_rent_by_state.png", dpi=150)
plt.close()
print("Plot 3: Avg rent by state saved")


# PLOT 4: Price by number of bedrooms (box plot)
# Keep 0–5 beds (covers the vast majority of listings)
beds_data = merged[merged["beds"].between(0, 5)].copy()
beds_data["beds"] = beds_data["beds"].astype(int)

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=beds_data, x="beds", y="price",
            palette="Blues", ax=ax, showfliers=False)
ax.set_title("Rental Price by Number of Bedrooms", fontsize=14, fontweight="bold")
ax.set_xlabel("Bedrooms")
ax.set_ylabel("Monthly Rent ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_price_by_beds.png", dpi=150)
plt.close()
print("Plot 4: Price by bedrooms saved")


# PLOT 5: Correlation heatmap
feature_cols = [
    "price", "sqfeet", "beds", "baths",
    "price_per_sqft", "median_income",
    "unemployment_rate", "bachelors_rate", "median_rent_census",
]
corr = merged[feature_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=ax,
    annot_kws={"size": 9}
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_correlation_heatmap.png", dpi=150)
plt.close()
print("Plot 5: Correlation heatmap saved")


# PLOT 6: Price vs Median Household Income
city_avg = (
    merged.groupby(["region", "state"])
    .agg(avg_price=("price", "mean"), median_income=("median_income", "first"))
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(city_avg["median_income"], city_avg["avg_price"],
           alpha=0.5, s=20, color="#004BC3")
ax.set_title("City Avg Rent vs. Median Household Income", fontsize=14, fontweight="bold")
ax.set_xlabel("Median Household Income ($)")
ax.set_ylabel("Average Monthly Rent ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_rent_vs_income.png", dpi=150)
plt.close()
print("Plot 6: Rent vs income saved")


print(f"\nAll plots saved to '{OUTPUT_DIR}/' folder")
print(f"  Median rent:          ${merged['price'].median():,.0f}")
corr_with_price = corr["price"].drop("price").sort_values(key=abs, ascending=False)
print(f"  Strongest correlates with price:")
for feat, val in corr_with_price.items():
    print(f"    {feat:<25} r = {val:.3f}")