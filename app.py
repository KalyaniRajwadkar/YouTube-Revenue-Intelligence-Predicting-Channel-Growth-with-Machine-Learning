
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------
# FIND PROJECT FILES
# ---------------------------------

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

MODEL_PATH = BASE_DIR / "youtube_revenue_model.pkl"
DATA_PATH = BASE_DIR / "youtube_cleaned_data.csv"

# Fallback: files may be in the same folder as app.py
if not MODEL_PATH.exists():
    MODEL_PATH = APP_DIR / "youtube_revenue_model.pkl"

if not DATA_PATH.exists():
    DATA_PATH = APP_DIR / "youtube_cleaned_data.csv"

# ---------------------------------
# PAGE SETTINGS
# ---------------------------------

st.set_page_config(
    page_title="YouTube Revenue Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Revenue Intelligence")
st.subheader("Channel Analytics & Estimated Revenue Prediction")
st.caption("Machine Learning Dashboard using Random Forest Regression")

# ---------------------------------
# LOAD MODEL AND DATASET
# ---------------------------------

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Check the location of youtube_revenue_model.pkl.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.info("Check the location of youtube_cleaned_data.csv.")
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, encoding="latin1")
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

required_keys = ["model", "features", "encoded_columns"]
if not isinstance(saved, dict) or not all(
    key in saved for key in required_keys
):
    st.error(
        "The saved model has an unexpected format. "
        "Please run train_model.py from the folder containing "
        "the training script and save the model again."
    )
    st.stop()

model = saved["model"]
features = saved["features"]
encoded_columns = saved["encoded_columns"]

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

st.success("Dataset and trained model loaded successfully!")

# ---------------------------------
# DATASET OVERVIEW
# ---------------------------------

st.header("📈 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Channels", f"{len(df):,}")
col2.metric("Model Features", len(features))
col3.metric("Dataset Columns", len(df.columns))

st.divider()

# ---------------------------------
# CHANNEL ANALYTICS
# ---------------------------------

st.header("🔎 Channel Analytics")

if "subscribers" in df.columns:
    df["subscribers"] = pd.to_numeric(
        df["subscribers"], errors="coerce"
    )

if "video_views" in df.columns:
    df["video_views"] = pd.to_numeric(
        df["video_views"], errors="coerce"
    )

if (
    "youtuber" in df.columns
    and "subscribers" in df.columns
):
    top = (
        df[["youtuber", "subscribers"]]
        .dropna()
        .nlargest(10, "subscribers")
        .sort_values("subscribers")
    )

    if not top.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top["youtuber"], top["subscribers"])
        ax.set_title("Top 10 Channels by Subscribers")
        ax.set_xlabel("Subscribers")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------
# REVENUE PREDICTION
# ---------------------------------

st.header("🤖 Estimated Monthly Revenue Prediction")
st.write("Enter channel details to estimate monthly revenue.")

numeric_features = [
    "subscribers",
    "video_views",
    "uploads",
    "subscribers_for_last_30_days",
    "video_views_for_the_last_30_days",
    "created_year"
]

with st.form("prediction_form"):
    user_input = {}
    left, right = st.columns(2)

    for i, col in enumerate(features):
        container = left if i % 2 == 0 else right

        with container:
            if col in numeric_features:
                if col in df.columns:
                    values = pd.to_numeric(
                        df[col], errors="coerce"
                    )
                    median = values.median()
                    default = (
                        float(median)
                        if pd.notna(median) else 0.0
                    )
                else:
                    default = 0.0

                user_input[col] = st.number_input(
                    col.replace("_", " ").title(),
                    min_value=0.0,
                    value=max(0.0, default),
                    step=1.0
                )
            else:
                if col in df.columns:
                    options = sorted(
                        df[col].dropna().astype(str).unique().tolist()
                    )
                else:
                    options = []

                if options:
                    user_input[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options
                    )
                else:
                    user_input[col] = st.text_input(
                        col.replace("_", " ").title()
                    )

    submitted = st.form_submit_button(
        "Predict Estimated Revenue"
    )

if submitted:
    try:
        input_df = pd.DataFrame([user_input])

        for col in numeric_features:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(
                    input_df[col], errors="coerce"
                )

        encoded_input = pd.get_dummies(
            input_df, dummy_na=True, dtype=int
        )

        encoded_input = encoded_input.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        prediction = float(model.predict(encoded_input)[0])

        if np.isfinite(prediction):
            st.subheader("Prediction Result")
            st.metric(
                "Estimated Monthly Revenue",
                f"${prediction:,.2f} USD"
            )
            st.info(
                "This is an estimate based on the dataset's "
                "earnings estimates. It is not verified actual "
                "revenue or a guarantee of future earnings."
            )
        else:
            st.error("The model returned an invalid prediction.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------
# FEATURE IMPORTANCE
# ---------------------------------

st.header("📊 Important Prediction Features")

try:
    regressor = model.named_steps["regressor"]

    importance = pd.Series(
        regressor.feature_importances_,
        index=encoded_columns
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    importance.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Features Used by the Model")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

except Exception as e:
    st.warning(f"Feature importance could not be displayed: {e}")

# ---------------------------------
# DATA PREVIEW
# ---------------------------------

with st.expander("View Dataset Preview"):
    st.dataframe(df.head(20), use_container_width=True)

st.caption(
    "Project: YouTube Revenue Intelligence | "
    "Python, Pandas, Scikit-learn & Streamlit"
)