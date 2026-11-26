from pathlib import Path


import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="📘",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "student-mat.csv"
FEATURES = ["studytime", "G1", "G2"]
TARGET = "G3"


@st.cache_data
def load_and_train(data_file: str, modified_time: float):
    """Load the class dataset, prepare the selected columns, and train one model."""
    data = pd.read_csv(data_file, sep=";")
    data.columns = data.columns.str.strip()
    data = data.drop_duplicates()
    clean_data = data[FEATURES + [TARGET]].dropna().copy()
    clean_data = clean_data[
        clean_data["studytime"].between(1, 4)
        & clean_data["G1"].between(0, 20)
        & clean_data["G2"].between(0, 20)
        & clean_data["G3"].between(0, 20)
    ]

    X = clean_data[FEATURES]
    y = clean_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, clean_data, y_test, predictions, scores


st.title("📘 Student Grade Predictor")
st.write(
    "Explore a simple linear regression model that estimates a student's final "
    "math grade from study time and the first two period grades."
)

if not DATA_FILE.exists():
    st.error("I can't find student-mat.csv beside app.py.")
    st.info(
        "Put the dataset file in the same folder as app.py, then refresh this page. "
        "The dataset uses semicolons between columns."
    )
    st.stop()

try:
    model, data, y_test, predictions, scores = load_and_train(
        str(DATA_FILE), DATA_FILE.stat().st_mtime
    )
except Exception as error:
    st.error("The dataset could not be loaded. Check that it is the UCI student-mat.csv file.")
    st.caption(f"Details: {error}")
    st.stop()

st.divider()
left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Try a student example")
    st.caption("Choose values, then press the button to get an estimate.")
    with st.form("prediction_form"):
        studytime = st.select_slider(
            "Weekly study time category",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda value: {
                1: "1 · Under 2 hours",
                2: "2 · 2 to 5 hours",
                3: "3 · 5 to 10 hours",
                4: "4 · Over 10 hours",
            }[value],
        )
        g1 = st.slider("First period grade (G1)", min_value=0, max_value=20, value=10)
        g2 = st.slider("Second period grade (G2)", min_value=0, max_value=20, value=11)
        submitted = st.form_submit_button("Estimate final grade", type="primary")

with right:
    st.subheader("Model check")
    metric1, metric2 = st.columns(2)
    metric1.metric("Mean absolute error", f"{scores['mae']:.2f} points")
    metric2.metric("R² score", f"{scores['r2']:.3f}")
    st.caption(
        "These scores come from a held-back test group. Lower mean error is better; "
        "R² describes how well predictions fit that group."
    )

if submitted:
    example = pd.DataFrame(
        {"studytime": [studytime], "G1": [g1], "G2": [g2]}, columns=FEATURES
    )
    estimate = float(model.predict(example)[0])
    # Grades are on a 0–20 scale; keep the displayed estimate within that scale.
    estimate = max(0.0, min(20.0, estimate))
    st.success(f"Estimated final grade: **{estimate:.1f} / 20**")
    st.caption(
        "This is a learning example, not a real grade or a guarantee. The model uses "
        "patterns in a small historical dataset."
    )

st.divider()
st.subheader("How close are the model's predictions?")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(y_test, predictions, alpha=0.65, color="#3478c7", edgecolors="white")
ax.plot([0, 20], [0, 20], linestyle="--", color="#e05b52", label="Perfect prediction")
ax.set(xlabel="Actual final grade (G3)", ylabel="Predicted final grade", xlim=(0, 20), ylim=(0, 20))
ax.grid(alpha=0.2)
ax.legend()
st.pyplot(fig, use_container_width=True)

with st.expander("What does the model use?"):
    st.write(
        "The app trains one linear regression using `studytime`, `G1`, and `G2` "
        "to predict `G3`. It removes duplicate rows, keeps those four columns, "
        "drops missing values, and keeps grades in the 0–20 range."
    )
    st.write(f"Rows used after cleaning: {len(data)}")
