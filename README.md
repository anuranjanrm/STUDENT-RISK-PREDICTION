# Student Academic Risk Prediction

**DCS & GDG AI/ML Recruitment Task** — an end-to-end machine learning project that
identifies students who are **at risk** and need academic intervention, and
automatically generates a risk report with recommendations.

## 📊 Dataset

[Student Dataset — Kaggle](https://www.kaggle.com/datasets/ganeshkumarofficial/student-dataset)

Features include demographics (gender, age, department), **Attendance (%)**,
assessment components (Midterm, Final, Assignments, Quizzes, Participation,
Projects), **Total_Score**, **Grade**, test-preparation-course status and
subject scores (math / reading / writing / science).

## 🎯 Project Pipeline

### 1. Data Exploration
- Distributions of all numeric features; categorical breakdowns
- **Attendance ↔ marks analysis**: scatter + regression line, Pearson
  correlation of attendance with every score component
- Full correlation heatmap; effect of the test-preparation course;
  per-department performance

### 2. Data Preprocessing
- Removed duplicates; clipped impossible values (scores/attendance → 0–100,
  age sanity check)
- Missing values → median (numeric) / mode (categorical)
- One-hot encoding of categorical features
- **Leakage control:** dropped identifiers (ID, names, email) and
  `Total_Score`, `Grade`, `Final_Score` — the first two *define* the label and
  the last one wouldn't exist yet at intervention time

### 3. Risk Definition & Models
Transparent, explainable risk criteria (thresholds match common academic cut-offs):

| Risk | Criteria | Action |
|---|---|---|
| 🔴 **High** | Total_Score < 60 **or** Attendance < 60 | Immediate intervention |
| 🟡 **Medium** | Total_Score or Attendance in [60, 75) | Monitor & support |
| 🟢 **Low** | Otherwise | Maintain habits |

Models trained (stratified 80/20 split, 5-fold cross-validation):
- Logistic Regression (with feature scaling)
- Decision Tree (`max_depth=6`)
- Random Forest (200 trees)

### 4. Evaluation
Accuracy, Precision, Recall, F1-score (weighted) and confusion matrices for all
models + full classification report for the best one (typically Random Forest).

### 5. Prediction & Recommendation
Produces the required output format per student, e.g.:

```
Student: B
Attendance: 62%
Marks: 55
Risk: HIGH
Recommendation: Improve attendance urgently (below 60%); focus on upcoming assessments and revision.
```

### 6. Automation (Bonus)
Automatically generates **`student_risk_report.csv`** for every student:

| Student | Attendance | Risk | Recommendation |
|---|---|---|---|

## 📁 Repository Contents

| File | Description |
|---|---|
| `student_risk_analysis.ipynb` | Main notebook — full analysis, models, evaluation |
| `student_risk_analysis.py` | Same pipeline as a plain Python script |
| `student_risk_report.csv` | Generated risk report (output of the notebook) |

## ▶️ How to Run

**Google Colab (recommended):**
1. Download the dataset from Kaggle (ZIP or extracted CSV)
2. Upload `student_risk_analysis.ipynb` to [colab.research.google.com](https://colab.research.google.com)
3. `Runtime → Run all` — upload the ZIP/CSV when prompted
4. `student_risk_report.csv` downloads automatically at the end

**Locally:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook student_risk_analysis.ipynb
# place the dataset CSV (or ZIP) next to the notebook first
```

## 🛠️ Tech Stack
Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn

## 💡 Key Findings
- Attendance is **positively correlated** with total score — a practical
  early-warning indicator
- Completing the test-preparation course is associated with higher average scores
- Component assessments (midterm, assignments, participation) all track the
  final outcome, enabling risk prediction **before** end-of-term

## ⚠️ Limitations & Future Work
- Risk thresholds (60/75) are heuristic — could be tuned with academic staff
- The label derives from Total_Score, so the model partially re-learns the rule;
  a future iteration could use next-term outcomes as a fully independent label
