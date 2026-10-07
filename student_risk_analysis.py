# %% [markdown]
# # Student Academic Risk Prediction
# **DCS & GDG AI/ML Recruitment Task**
#
# **Goal:** explore the student dataset, build an ML model that flags students
# who are *at risk* and need academic intervention, evaluate it, and generate
# an automated risk report.
#
# **Dataset:** https://www.kaggle.com/datasets/ganeshkumarofficial/student-dataset
#
# **Approach in one paragraph:**
# I first explore the data (with focus on the attendance ↔ marks relationship),
# then clean and encode it. I define a transparent, rule-based risk label
# (Low / Medium / High) from *overall performance and attendance*, then train
# three classifiers (Logistic Regression, Decision Tree, Random Forest) on
# **features that would realistically be available before the end of the term**
# (i.e. excluding Total_Score, Grade and Final_Score, which either define the
# label or wouldn't exist yet at intervention time). The best model is then
# used to produce per-student recommendations and a full automated report.

# %%
# ============================ 0. SETUP & DATA LOADING ============================
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
pd.set_option("display.max_columns", None)

# ---- Load the dataset ----
# Kaggle's "Download" button gives you a ZIP (usually "archive.zip") with the CSV inside.
# This cell accepts EITHER the ZIP directly or the extracted CSV.
import zipfile

def extract_zips():
    for z in glob.glob("*.zip"):
        print(f"Extracting {z} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(".")

try:
    from google.colab import files
    if not glob.glob("*.csv") and not glob.glob("*.zip"):
        print("Upload the dataset ZIP (or CSV) downloaded from Kaggle:")
        files.upload()
except ImportError:
    pass

extract_zips()   # no-op if you uploaded the CSV directly

csv_files = sorted(glob.glob("*.csv"))
assert csv_files, "No CSV file found — upload the Kaggle ZIP or the extracted CSV."
print("Loading:", csv_files[0])
df = pd.read_csv(csv_files[0])

# ---- Normalize column names ----
# Kaggle CSV headers often contain spaces / brackets / '%' (e.g. "Attendance (%)").
# Normalizing to lowercase_underscore makes the rest of the code robust.
df.columns = (df.columns.str.strip().str.lower()
                        .str.replace(r"[^a-z0-9]+", "_", regex=True)
                        .str.strip("_"))

def find_col(frame, *keywords):
    """Return the first column containing ALL given keywords (fuzzy, robust to naming)."""
    for c in frame.columns:
        if all(k in c for k in keywords):
            return c
    return None

# Locate the key columns (falling back gracefully if a name differs slightly)
COL_ID     = find_col(df, "student", "id") or find_col(df, "id")
COL_FIRST  = find_col(df, "first")
COL_LAST   = find_col(df, "last")
COL_EMAIL  = find_col(df, "email")
COL_GENDER = find_col(df, "gender")
COL_AGE    = find_col(df, "age")
COL_DEPT   = find_col(df, "department")
COL_ATT    = find_col(df, "attendance")
COL_MID    = find_col(df, "midterm")
COL_FINAL  = find_col(df, "final")
COL_ASSIGN = find_col(df, "assignment")
COL_QUIZ   = find_col(df, "quiz")
COL_PART   = find_col(df, "participation")
COL_PROJ   = find_col(df, "project")
COL_TOTAL  = find_col(df, "total")
COL_GRADE  = find_col(df, "grade")
COL_PREP   = find_col(df, "preparation") or find_col(df, "prep")

print("Detected columns:", df.columns.tolist())
df.head()

# %% [markdown]
# ## 1. Data Exploration
# First look at structure, types, missing values, duplicates and basic stats.

# %%
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
df.describe(include="all").T

# %%
# --- Distributions of the numeric performance columns ---
# Histograms reveal skew, outliers and impossible values (e.g. scores > 100).
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols].hist(bins=20, figsize=(14, 10), edgecolor="black")
plt.suptitle("Distributions of numeric features", y=1.02)
plt.tight_layout()
plt.show()

# %%
# --- Categorical overview: gender, department, grade, prep course ---
for col in [COL_GENDER, COL_DEPT, COL_GRADE, COL_PREP]:
    if col:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index,
                      palette="viridis")
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ### 1.1 The key question: how does **attendance** relate to **marks**?
# We check this three ways:
# 1. Scatter + regression line of attendance vs total score
# 2. Pearson correlation of attendance with every score component
# 3. Full correlation heatmap to spot which factors drive performance overall

# %%
# Scatter plot with regression line — the single most important plot of the EDA.
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x=COL_ATT, y=COL_TOTAL, scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"})
plt.title("Attendance vs Total Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Total Score")
plt.tight_layout()
plt.show()

corr_att_total = df[COL_ATT].corr(df[COL_TOTAL])
print(f"Pearson correlation between attendance and total score: {corr_att_total:.3f}")

# %%
# Correlation of attendance with every individual assessment component
score_components = [c for c in [COL_MID, COL_FINAL, COL_ASSIGN, COL_QUIZ,
                                COL_PART, COL_PROJ, COL_TOTAL,
                                find_col(df, "math"), find_col(df, "reading"),
                                find_col(df, "writing"), find_col(df, "science")] if c]
att_corr = df[[COL_ATT] + score_components].corr()[COL_ATT].drop(COL_ATT).sort_values()
print("Correlation of Attendance with each score component:\n", att_corr.round(3))

att_corr.plot(kind="barh", color="steelblue", edgecolor="black")
plt.title("How strongly does attendance correlate with each score?")
plt.xlabel("Pearson correlation with Attendance")
plt.tight_layout()
plt.show()

# %%
# Full correlation heatmap of numeric features
plt.figure(figsize=(11, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Correlation heatmap of numeric features")
plt.tight_layout()
plt.show()

# %%
# Does the test preparation course help? Compare average total scores.
if COL_PREP:
    df.groupby(COL_PREP)[COL_TOTAL].mean().plot(kind="bar", color="seagreen",
                                                edgecolor="black")
    plt.title("Average Total Score by Test Preparation Course status")
    plt.ylabel("Average Total Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Average total score per department — are some departments struggling?
if COL_DEPT:
    df.groupby(COL_DEPT)[COL_TOTAL].mean().sort_values().plot(
        kind="barh", color="steelblue", edgecolor="black")
    plt.title("Average Total Score by Department")
    plt.xlabel("Average Total Score")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Key EDA takeaways** (auto-computed below so the numbers are always true for
# whatever version of the CSV is used):
# * Attendance shows a clear **positive correlation** with total score —
#   students who attend more, score more.
# * Component scores (midterm, assignments, quizzes, participation, projects)
#   all correlate positively with the total score, as expected.
# * Completing the test-preparation course is associated with higher averages.

# %%
print(f"Attendance ↔ Total Score correlation: {corr_att_total:.3f}")
if corr_att_total > 0.3:
    print("→ Strong positive relationship: attendance is a meaningful risk indicator.")
elif corr_att_total > 0.1:
    print("→ Moderate positive relationship: attendance matters but is not the whole story.")
else:
    print("→ Weak linear relationship: other factors dominate performance.")

# %% [markdown]
# ## 2. Data Preprocessing
# Steps:
# 1. **Duplicates** — drop exact duplicate rows.
# 2. **Impossible / incorrect values** — scores and attendance must be within
#    0–100; ages outside 10–100 are treated as errors → set to NaN, then imputed.
# 3. **Missing values** — numeric → median (robust to outliers),
#    categorical → mode.
# 4. **Encoding** — one-hot encode Gender / Department / prep-course.
# 5. **Feature selection** — drop identifiers (ID, names, email) and the
#    leakage columns (Total_Score, Grade define the label; Final_Score would
#    not yet exist when we want to intervene).

# %%
df_clean = df.copy()

# 1. duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Dropped {before - len(df_clean)} duplicate rows.")

# 2. impossible values: clip every score/attendance-like column to [0, 100]
score_like = [c for c in df_clean.columns
              if any(k in c for k in ["score", "avg", "attendance"])]
for c in score_like:
    bad = ((df_clean[c] < 0) | (df_clean[c] > 100)).sum()
    if bad:
        print(f"Column '{c}': {bad} out-of-range values clipped to [0,100].")
    df_clean[c] = df_clean[c].clip(0, 100)

# age sanity check
if COL_AGE:
    bad_age = ((df_clean[COL_AGE] < 10) | (df_clean[COL_AGE] > 100)).sum()
    if bad_age:
        print(f"Column '{COL_AGE}': {bad_age} impossible ages set to NaN.")
        df_clean.loc[(df_clean[COL_AGE] < 10) | (df_clean[COL_AGE] > 100), COL_AGE] = np.nan

# 3. missing values: median for numeric, mode for categorical
medians, modes = {}, {}
for c in df_clean.columns:
    if df_clean[c].dtype != object:
        medians[c] = df_clean[c].median()
        df_clean[c] = df_clean[c].fillna(medians[c])
    else:
        modes[c] = df_clean[c].mode()[0]
        df_clean[c] = df_clean[c].fillna(modes[c])

print("\nMissing values after cleaning:", int(df_clean.isna().sum().sum()))

# %% [markdown]
# ### 2.1 Define the risk label — *Low / Medium / High*
#
# **My criteria** (transparent business rule, easy to explain to staff):
# * **High risk** — Total_Score < 60 **or** Attendance < 60
#   *(failing grades and/or critically low attendance → immediate intervention)*
# * **Medium risk** — Total_Score in [60, 75) **or** Attendance in [60, 75)
#   *(borderline — monitor and support)*
# * **Low risk** — everyone else.
#
# The thresholds 60 / 75 match common academic cut-offs (pass mark ≈ 60%,
# satisfactory ≈ 75%) and the EDA finding that attendance tracks performance.

# %%
HIGH, MEDIUM, LOW = 2, 1, 0   # numeric codes keep confusion matrices ordered
RISK_NAMES = {LOW: "LOW", MEDIUM: "MEDIUM", HIGH: "HIGH"}

def risk_label(row):
    total, att = row[COL_TOTAL], row[COL_ATT]
    if total < 60 or att < 60:
        return HIGH
    if total < 75 or att < 75:
        return MEDIUM
    return LOW

df_clean["risk"] = df_clean.apply(risk_label, axis=1)
print("Risk class distribution:")
print(df_clean["risk"].map(RISK_NAMES).value_counts())
print("\nPercentages:")
print((df_clean["risk"].map(RISK_NAMES).value_counts(normalize=True) * 100).round(1))

# %%
# 4 & 5. Encoding + feature selection
cat_cols = [c for c in [COL_GENDER, COL_DEPT, COL_PREP] if c]
model_df = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

# Columns the model must NOT see:
#  - identifiers: no predictive value, just leak identity
#  - total_score / grade: they *define* the label → target leakage
#  - final_score: at intervention time the final exam hasn't happened yet
drop_cols = {c for c in [COL_ID, COL_FIRST, COL_LAST, COL_EMAIL,
                         COL_TOTAL, COL_GRADE, COL_FINAL] if c}
feature_cols = [c for c in model_df.columns if c not in drop_cols and c != "risk"]

X = model_df[feature_cols]
y = model_df["risk"]
print(f"Using {len(feature_cols)} features:", feature_cols)

# %% [markdown]
# ## 3. ML Models
# Train three standard classifiers with a stratified 80/20 split
# (stratify keeps the Low/Medium/High proportions identical in train & test):
# * **Logistic Regression** — interpretable linear baseline (needs scaling)
# * **Decision Tree** — transparent rules, mirrors how staff think
# * **Random Forest** — ensemble, usually the strongest of the three

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),          # LR is scale-sensitive
        ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, m in models.items():
    m.fit(X_train, y_train)
    # 5-fold CV on the training set gives a more honest estimate than one split
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"{name:20s} trained | 5-fold CV weighted-F1 = {cv.mean():.3f} ± {cv.std():.3f}")

# %% [markdown]
# ## 4. Evaluation
# Metrics: **Accuracy, Precision, Recall, F1-score (weighted) and the Confusion
# Matrix**. Weighted averaging is used because the risk classes are imbalanced;
# the per-class breakdown is shown in the classification report.

# %%
results = {}
for name, m in models.items():
    y_pred = m.predict(X_test)
    results[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score":  f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

results_df = pd.DataFrame(results).T.round(3)
print("=== Model comparison (test set) ===")
print(results_df)

# %%
# Confusion matrix for every model, side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = [RISK_NAMES[i] for i in sorted(RISK_NAMES)]
for ax, (name, m) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, m.predict(X_test), labels=sorted(RISK_NAMES))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", y=1.02)
plt.tight_layout()
plt.show()

# %%
best_name = results_df["F1-score"].idxmax()
best_model = models[best_name]
print(f"Best model by weighted F1: {best_name}\n")
print(classification_report(y_test, best_model.predict(X_test),
                            target_names=labels, zero_division=0))

# %%
# Feature importances (Random Forest) — which factors drive the risk prediction?
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
imp.tail(12).plot(kind="barh", color="darkorange", edgecolor="black")
plt.title("Top feature importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Prediction & Recommendation
# A helper that turns one student's data into the exact output format required:
# ```
# Student: B
# Attendance: 62%
# Marks: 55
# Risk: HIGH
# Recommendation: Improve attendance and focus on upcoming assessments.
# ```

# %%
def make_recommendation(att, mid, assign, part, prep_done):
    """Rule-based, human-readable advice targeting the student's weakest areas."""
    tips = []
    if att < 60:
        tips.append("improve attendance urgently (below 60%)")
    elif att < 75:
        tips.append("improve attendance consistency")
    if mid is not None and mid < 60:
        tips.append("focus on upcoming assessments and revision")
    if assign is not None and assign < 60:
        tips.append("complete and submit assignments on time")
    if part is not None and part < 60:
        tips.append("participate more actively in class")
    if prep_done is not None and not prep_done:
        tips.append("enroll in the test preparation course")
    if not tips:
        return "Keep up the good work and maintain current study habits."
    return "; ".join(tips).capitalize() + "."

def student_features(raw_row):
    """Convert one row of the ORIGINAL dataframe into a model-ready vector,
    applying the same imputation/encoding used in training."""
    r = raw_row.copy()
    for c, v in medians.items():
        if c in r and pd.isna(r[c]):
            r[c] = v
    tmp = pd.DataFrame([r])
    tmp = pd.get_dummies(tmp, columns=[c for c in cat_cols if c in tmp], drop_first=True)
    return tmp.reindex(columns=feature_cols, fill_value=0)

def predict_student(raw_row):
    """Print the required output block for one student."""
    name = " ".join(str(raw_row.get(c, "")) for c in [COL_FIRST, COL_LAST] if c).strip() \
           or str(raw_row.get(COL_ID, "Unknown"))
    att, total = raw_row[COL_ATT], raw_row[COL_TOTAL]
    risk_code = best_model.predict(student_features(raw_row))[0]
    prep_done = None
    if COL_PREP and COL_PREP in raw_row:
        prep_done = str(raw_row[COL_PREP]).strip().lower() in ("completed", "yes", "1", "true")
    rec = make_recommendation(att, raw_row.get(COL_MID), raw_row.get(COL_ASSIGN),
                              raw_row.get(COL_PART), prep_done)
    print(f"Student: {name}")
    print(f"Attendance: {att:.0f}%")
    print(f"Marks: {total:.0f}")
    print(f"Risk: {RISK_NAMES[risk_code]}")
    print(f"Recommendation: {rec}")
    print("-" * 50)

# Demo: run the required-format output for a few students from the dataset
for _, row in df_clean.sample(5, random_state=1).iterrows():
    predict_student(row)

# %% [markdown]
# ## 6. Automation (Bonus) — automated risk report
# Generates **Student | Attendance | Risk | Recommendation** for every student
# and exports it to `student_risk_report.csv` (auto-downloaded in Colab).

# %%
report_rows = []
for _, row in df_clean.iterrows():
    name = " ".join(str(row.get(c, "")) for c in [COL_FIRST, COL_LAST] if c).strip() \
           or str(row.get(COL_ID, "Unknown"))
    risk_code = best_model.predict(student_features(row))[0]
    prep_done = None
    if COL_PREP and COL_PREP in row:
        prep_done = str(row[COL_PREP]).strip().lower() in ("completed", "yes", "1", "true")
    report_rows.append({
        "Student": name,
        "Attendance": f"{row[COL_ATT]:.0f}%",
        "Risk": RISK_NAMES[risk_code],
        "Recommendation": make_recommendation(
            row[COL_ATT], row.get(COL_MID), row.get(COL_ASSIGN),
            row.get(COL_PART), prep_done),
    })

report_df = pd.DataFrame(report_rows)
report_df.to_csv("student_risk_report.csv", index=False)
print(f"Report generated for {len(report_df)} students → student_risk_report.csv")
print(report_df["Risk"].value_counts())
report_df.head(10)

# Auto-download when running in Google Colab
try:
    from google.colab import files
    files.download("student_risk_report.csv")
except ImportError:
    pass

# %% [markdown]
# ## Conclusion
# * **Attendance is positively correlated with marks** — it is a legitimate,
#   easy-to-measure early-warning indicator (see Section 1.1).
# * The risk rule (performance + attendance thresholds at 60/75) is simple to
#   explain, and the ML model generalises it using only data available *before*
#   the term ends, so interventions can happen in time.
# * Of the three models, the one with the best weighted F1 (typically the
#   **Random Forest**) was selected for the final system.
# * The system outputs the required per-student block and a full automated
#   report (`student_risk_report.csv`) with actionable recommendations.
#
# **Limitations / future work:** the risk thresholds are heuristic (could be
# tuned with teachers), and the label derives from Total_Score, so the model
# partly re-learns the rule — a future iteration could use next-term results
# as a truly independent label.
