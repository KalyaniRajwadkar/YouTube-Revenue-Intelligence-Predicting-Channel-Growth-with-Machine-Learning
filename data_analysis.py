
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Load dataset
file_path = Path(__file__).parent / "Global YouTube Statistics.csv"
df = pd.read_csv(file_path, encoding="latin1")

# Step 2: Clean column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\n===== DATASET OVERVIEW =====")
print("Rows and columns:", df.shape)

# Step 3: Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum().sort_values(ascending=False))

# Step 4: Duplicate rows
print("\n===== DUPLICATE ROWS =====")
print("Duplicates:", df.duplicated().sum())

# Step 5: Remove duplicate rows
df = df.drop_duplicates()

# Step 6: Convert numeric columns
numeric_columns = [
    "subscribers",
    "video_views",
    "uploads",
    "lowest_monthly_earnings",
    "highest_monthly_earnings",
    "lowest_yearly_earnings",
    "highest_yearly_earnings",
    "subscribers_for_last_30_days",
    "video_views_for_the_last_30_days"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Step 7: Basic statistics
print("\n===== NUMERICAL SUMMARY =====")
print(df.describe().T)

# Step 8: Top 10 channels by subscribers
if "subscribers" in df.columns and "youtuber" in df.columns:
    top_channels = df.nlargest(10, "subscribers")

    print("\n===== TOP 10 CHANNELS =====")
    print(top_channels[["youtuber", "subscribers"]].to_string(index=False))

    plt.figure(figsize=(12, 6))
    plt.barh(top_channels["youtuber"], top_channels["subscribers"])
    plt.xlabel("Subscribers")
    plt.ylabel("YouTube Channel")
    plt.title("Top 10 YouTube Channels by Subscribers")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "top_channels.png")
    plt.show()

# Step 9: Correlation analysis
available_cols = [
    col for col in ["subscribers", "video_views", "uploads"]
    if col in df.columns
]

if len(available_cols) >= 2:
    print("\n===== CORRELATION MATRIX =====")
    print(df[available_cols].corr())

    df[available_cols] = df[available_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    df[available_cols].corr().plot(
        kind="bar",
        figsize=(10, 5),
        title="YouTube Channel Metrics Correlation"
    )
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "correlation.png")
    plt.show()

# Step 10: Save cleaned dataset
output_path = Path(__file__).parent / "youtube_cleaned_data.csv"
df.to_csv(output_path, index=False)

print("\n===== ANALYSIS COMPLETED =====")
print("Cleaned dataset saved to:", output_path)
