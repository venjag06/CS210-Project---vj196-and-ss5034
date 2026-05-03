# A Data-Driven Approach to Assessing Rental Price Fairness
### CS210 Project — Shreyansh Sen & Venkatesh Jagadesan

This project uses machine learning to predict fair rental prices for US housing listings and flag whether a given listing is overpriced or underpriced. It combines Craigslist rental data from Kaggle with US Census demographic data to train two regression models — Linear Regression and K-Nearest Neighbors — and exposes the predictions through a command-line tool where a user can enter a property and see how its price compares to model predictions.

---

## Data Sources

**Kaggle — USA Housing Listings** (`austinreese/usa-housing-listings`)
Craigslist rental listings scraped across the US. Contains ~380k rows with fields for region, price, property type, square footage, beds, baths, and state.

**US Census Bureau ACS 5-Year Estimates (2022)**
City-level demographic data fetched via the Census REST API for every city present in the Kaggle dataset. Variables used:
- `B19013_001E` — Median household income
- `B01003_001E` — Total population
- `B23025_005E` — Unemployed civilians
- `B15003_022E` — Bachelor's degree holders
- `B25064_001E` — Median gross rent

---

## Project Structure

```
├── load_data.py           # Downloads Kaggle data + fetches Census API data → housing.db
├── feature_engineering.py # Joins housing + cities, engineers new features → merged table
├── eda.py                 # Generates exploratory visualizations → eda_plots/
├── linear_regression.py   # Trains Linear Regression model → models/
├── knn_regression.py      # Trains KNN Regression model → models/
├── predict.py             # CLI tool: enter a property, get predicted fair price
├── models/                # Saved model files (generated after training)
├── eda_plots/             # Output charts (generated after running eda.py)
├── housing.db             # SQLite database (generated after running load_data.py)
└── .env                   # API keys — never committed to GitHub
```

---

## Setup

**1. Clone the repo and install dependencies:**
```bash
pip install pandas scikit-learn matplotlib seaborn kagglehub requests joblib python-dotenv
```

**2. Create a `.env` file** in the project root with your Census API key:
```
CENSUS_API_KEY=your_key_here
```
Get a free key at [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html).

**3. Set up Kaggle credentials** so `kagglehub` can download the dataset. Follow the [Kaggle API docs](https://www.kaggle.com/docs/api) to place your `kaggle.json` in `~/.kaggle/`.

---

## How to Run

Run the scripts in order. Each one caches its output to `housing.db`, so re-running is fast — it skips any step that's already been completed.

### Step 1 — Load data
```bash
python load_data.py
```
Downloads the Kaggle housing dataset and fetches Census data for every city in the dataset. Saves two tables to `housing.db`: `housing` (~324k rows after cleaning) and `cities` (330 cities matched to Census data).

### Step 2 — Feature engineering
```bash
python feature_engineering.py
```
Joins `housing` and `cities` on region + state, then engineers four new features:

| Feature | Description |
|---|---|
| `price_per_sqft` | Price divided by square footage |
| `avg_rent_by_region` | Mean rent for all listings in that city |
| `unemployment_rate` | Unemployed count as % of population |
| `bachelors_rate` | Bachelor's degree holders as % of population |

Saves the result as the `merged` table (~324k rows).

### Step 3 — Exploratory data analysis
```bash
python eda.py
```
Generates 6 plots saved to `eda_plots/`:
1. Distribution of rental prices
2. Price vs. square footage
3. Average rent by state
4. Price by number of bedrooms
5. Feature correlation heatmap
6. City average rent vs. median household income

### Step 4 — Train models
```bash
python linear_regression.py
python knn_regression.py
```
Both scripts use the same feature set and an 80/20 train/test split. Trained models are saved to `models/` so `predict.py` can load them without retraining.

**Features used:**
- `sqfeet`, `beds`, `baths` (property characteristics)
- `median_income`, `unemployment_rate`, `bachelors_rate`, `median_rent_census` (city demographics)
- `type` — one-hot encoded (apartment, house, condo, etc.)

The target variable is `log(price)` rather than raw price to correct for the right-skewed distribution of rents. Predictions are exponentiated back to dollars.

**KNN note:** Cross-validation over k = 3–25 is used to find the optimal number of neighbors. `weights="distance"` is used so closer neighbors have more influence than farther ones. Features are standardized before distance is computed.

### Step 5 — Run the predictor
```bash
python predict.py
```
Interactive CLI that asks for city, state, property details, and an optional listed price, then outputs predicted fair prices from both models and a verdict.

**Example output:**
```
══════════════════════════════════════════════════
       Rental Price Predictor
══════════════════════════════════════════════════
  City:          Austin, TX
  Property:      2bd / 1.0ba  |  900 sqft  |  apartment

  Linear Regression prediction:  $  1,183.00
  KNN prediction:                $  1,221.00
  Avg listed rent in this city:  $  1,247.00

  Your listed price:  $1,600.00

  vs LR prediction:              +$417.00  (+35.2%)  [OVER]
  vs KNN prediction:             +$379.00  (+31.0%)  [OVER]

  Verdict: Both models suggest this listing is OVERPRICED.
══════════════════════════════════════════════════
```

---

## Model Results

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | ~$350 | ~0.45 |
| KNN (k optimized via CV) | $268 | 0.82 |

KNN outperforms Linear Regression on this dataset because rental pricing has non-linear relationships — the effect of adding a bedroom on price varies significantly by city and price tier, which KNN handles naturally by finding locally similar listings.

---

## Key Findings from EDA

- Median US rent in the dataset: **$1,010/month**
- Most expensive states: **HI, MA, CA**
- Most affordable states: **KS, OK, MO**
- Strongest predictors of price: `price_per_sqft` (r=0.59), `median_income` (r=0.44), `sqfeet` (r=0.39)
- Unemployment rate and `median_rent_census` showed weak correlation with individual listing prices, likely because city-level averages don't capture within-city variation

---

## Limitations

- **City-level demographics only** — we don't have neighborhood, zip code, floor number, building amenities, or proximity to transit. These factors can explain large price differences within a single city.
- **Data age** — the Kaggle dataset is a historical Craigslist snapshot. Rents in many markets have changed significantly since it was collected.
- **City coverage** — 330 of ~427 unique Craigslist regions were successfully matched to Census data (77%). Unmatched cities are excluded from the model.
