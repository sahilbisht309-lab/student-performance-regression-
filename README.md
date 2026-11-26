# Student Grade Predictor

A beginner-friendly web app for the student performance regression project. It trains one linear regression model to estimate the final math grade (`G3`) from study time (`studytime`) and earlier grades (`G1` and `G2`).

## Project files

- `app.py` — the website and prediction logic.
- `student-mat.csv` — the dataset used by the app. Put it beside `app.py`.
- `requirements.txt` — Python packages the app needs.

## Run it in PyCharm on Windows

1. Put `student-mat.csv` in this folder, beside `app.py`.
2. Open this folder as a project in PyCharm.
3. Open PyCharm's terminal and install the packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the website:

   ```bash
   python -m streamlit run app.py
   ```

5. Streamlit will open the app in your browser. If it does not, use the local address shown in the terminal (usually `http://localhost:8501`).

## What the app shows

- Inputs for study time, first-period grade, and second-period grade.
- An estimated final grade on a 0–20 scale.
- Mean absolute error and R² from a held-back test group.
- A chart comparing actual and predicted grades.

This is a learning project using a historical student dataset. It is not intended to predict real students' grades or make decisions about students.
