# Student Performance Regression [Open the live Student Grade Predictor](https://student-grade-predictor0.streamlit.app/)

A beginner machine learning project that uses linear regression to predict students’ final math grades.

## What it does

The notebook loads and cleans the student data, trains a regression model, and compares predicted grades with actual grades.

- **Inputs:** study time (`studytime`), first-period grade (`G1`), and second-period grade (`G2`)
- **Prediction:** final math grade (`G3`)

## Files

- `studets.ipynb` — Jupyter notebook with the project code and results
- `student-mat.csv` — dataset used by the notebook

## Run the notebook

1. Install Python and Jupyter.
2. Install the required packages:

   ```bash
   python -m pip install pandas scikit-learn matplotlib
   ```

3. Keep `student-mat.csv` in the same folder as the notebook.
4. Open `studets.ipynb` in Jupyter or PyCharm and run the cells from top to bottom.

## Dataset

The dataset is the UCI Student Performance dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student%2Bperformance).
