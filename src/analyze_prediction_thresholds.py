import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------
# 1. Paths
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

train_path = BASE_DIR / "data" / "processed" / "train_features.csv"
test_path = BASE_DIR / "data" / "processed" / "test_features.csv"

# ---------------------------------
# 2. Load data
# ---------------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target = "Target_Next_Course_Type"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# ---------------------------------
# 3. Train balanced Random Forest
# ---------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------
# 4. Get Paid probabilities
# ---------------------------------
paid_probability = model.predict_proba(X_test)[:, 1]

# ---------------------------------
# 5. Evaluate different thresholds
# ---------------------------------
thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

results = []

for threshold in thresholds:

    y_pred = (
        paid_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Paid_Predictions": int(y_pred.sum()),
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

# ---------------------------------
# 6. Display results
# ---------------------------------
results_df = pd.DataFrame(results)

print("\n================================")
print("PREDICTION THRESHOLD ANALYSIS")
print("================================")

print(
    results_df.to_string(index=False)
)

# ---------------------------------
# 7. Best F1 threshold
# ---------------------------------
best_row = results_df.loc[
    results_df["F1_Score"].idxmax()
]

print("\n================================")
print("BEST F1 THRESHOLD")
print("================================")

print(
    f"Threshold       : {best_row['Threshold']:.2f}"
)

print(
    f"Paid Predictions: {int(best_row['Paid_Predictions'])}"
)

print(
    f"Precision       : {best_row['Precision']:.4f}"
)

print(
    f"Recall          : {best_row['Recall']:.4f}"
)

print(
    f"F1 Score        : {best_row['F1_Score']:.4f}"
)