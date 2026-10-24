
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "youtube_cleaned_data.csv"

if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "Global YouTube Statistics.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\n===== YOUTUBE REVENUE INTELLIGENCE =====")
print("Dataset loaded:", DATA_PATH.name)
print("Dataset shape:", df.shape)

# ==========================================
# STEP 2: PREPARE REVENUE TARGET
# ==========================================

required = [
    "lowest_monthly_earnings",
    "highest_monthly_earnings"
]

if not all(col in df.columns for col in required):
    raise ValueError(
        "Monthly earnings columns not found. "
        "Check the dataset column names."
    )

low = pd.to_numeric(
    df["lowest_monthly_earnings"], errors="coerce"
)
high = pd.to_numeric(
    df["highest_monthly_earnings"], errors="coerce"
)

# Midpoint of the published earnings range.
# This is an estimated target, not verified actual revenue.
df["estimated_monthly_revenue"] = (low + high) / 2

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["estimated_monthly_revenue"])

# ==========================================
# STEP 3: SELECT AVAILABLE FEATURES
# ==========================================

possible_features = [
    "subscribers",
    "video_views",
    "uploads",
    "subscribers_for_last_30_days",
    "video_views_for_the_last_30_days",
    "channel_type",
    "category",
    "country",
    "created_year"
]

features = [
    col for col in possible_features
    if col in df.columns
]

if len(features) == 0:
    raise ValueError("No suitable feature columns found.")

X = df[features].copy()
y = df["estimated_monthly_revenue"]

# Convert numeric columns safely
numeric_features = [
    "subscribers",
    "video_views",
    "uploads",
    "subscribers_for_last_30_days",
    "video_views_for_the_last_30_days",
    "created_year"
]

for col in numeric_features:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

# Encode categorical columns
X = pd.get_dummies(X, dummy_na=True, dtype=int)

# Remove unusable columns
X = X.dropna(axis=1, how="all")

# Remove rows with no usable feature values
valid_rows = X.notna().any(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows]

if len(X) < 10:
    raise ValueError(
        "Not enough usable rows to train the model."
    )

if X.shape[1] == 0:
    raise ValueError("No usable feature columns remain.")

print("\nFeatures used:", len(features))
print("Encoded model columns:", X.shape[1])
print("Training samples available:", len(X))

# ==========================================
# STEP 4: SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# STEP 5: TRAIN RANDOM FOREST
# ==========================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("Model training completed!")

# ==========================================
# STEP 6: EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE:  ${mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

# ==========================================
# STEP 7: FEATURE IMPORTANCE
# ==========================================

rf = model.named_steps["regressor"]

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n===== TOP 15 IMPORTANT FEATURES =====")
print(importance.head(15).to_string())

importance.head(15).sort_values().plot(
    kind="barh",
    figsize=(10, 6),
    title="Top Features for Estimated YouTube Revenue"
)

plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig(BASE_DIR / "feature_importance.png")
plt.close()

# ==========================================
# STEP 8: ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7)

min_value = min(y_test.min(), predictions.min())
max_value = max(y_test.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Estimated Revenue from Dataset (USD)")
plt.ylabel("Model Predicted Revenue (USD)")
plt.title("Actual Proxy vs Predicted Monthly Revenue")
plt.tight_layout()
plt.savefig(BASE_DIR / "revenue_prediction.png")
plt.close()

# ==========================================
# STEP 9: SAVE MODEL
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": features,
        "encoded_columns": X.columns.tolist(),
        "target": "estimated_monthly_revenue"
    },
    BASE_DIR / "youtube_revenue_model.pkl"
)

# Save test predictions for inspection
results = pd.DataFrame({
    "estimated_revenue_usd": y_test,
    "predicted_revenue_usd": predictions
})

results.to_csv(
    BASE_DIR / "revenue_predictions.csv",
    index=False
)

print("\n===== PROJECT OUTPUTS SAVED =====")
print("1. youtube_revenue_model.pkl")
print("2. feature_importance.png")
print("3. revenue_prediction.png")
print("4. revenue_predictions.csv")
print("\nMachine Learning pipeline completed!")
