
import pandas as pd
from pathlib import Path

# Step 1: Load dataset
file_path = Path(__file__).parent / "Global YouTube Statistics.csv"

try:
    df = pd.read_csv(file_path, encoding="latin1")

    print("\n===== DATASET LOADED SUCCESSFULLY =====")
    print("Total Rows and Columns:", df.shape)

    # Step 2: Display column names
    print("\n===== COLUMN NAMES =====")
    print(df.columns.tolist())

    # Step 3: Display first 5 rows
    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    # Step 4: Dataset information
    print("\n===== DATASET INFORMATION =====")
    df.info()

    # Step 5: Missing values
    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    # Step 6: Duplicate rows
    print("\n===== DUPLICATE ROWS =====")
    print(df.duplicated().sum())

    # Step 7: Statistical summary
    print("\n===== STATISTICAL SUMMARY =====")
    print(df.describe(include="all").T)

    # Step 8: Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print("\n===== CLEANED COLUMN NAMES =====")
    print(df.columns.tolist())

    print("\nDataset inspection completed!")

except FileNotFoundError:
    print("ERROR: CSV file not found!")
    print("Keep Global YouTube Statistics.csv in the same folder as load_data.py.")

except Exception as e:
    print("An error occurred:", e)
