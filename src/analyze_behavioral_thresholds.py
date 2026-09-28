import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------
# 1. Paths
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

sequential_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_sequential_dataset.csv"
)

# ---------------------------------
# 2. Load data
# ---------------------------------
df = pd.read_csv(sequential_path)

df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"]
)

# ---------------------------------
# 3. Calculate previous paid ratio
# ---------------------------------
df["Previous_Paid_Ratio"] = (
    df["Previous_Paid_Count"]
    / df["Previous_Transaction_Count"]
)

# ---------------------------------
# 4. Chronological ordering
# ---------------------------------
df = df.sort_values(
    "TransactionDate"
).reset_index(drop=True)

# ---------------------------------
# 5. Use final 20% as test data
# ---------------------------------
split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:].copy()

y_test = test_df["Target_Next_Course_Type"]

# ---------------------------------
# 6. Test different behavioral thresholds
# ---------------------------------
thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []

for threshold in thresholds:

    y_pred = (
        test_df["Previous_Paid_Ratio"] >= threshold
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
# 7. Display results
# ---------------------------------
results_df = pd.DataFrame(results)

print("\n================================")
print("BEHAVIORAL THRESHOLD ANALYSIS")
print("================================")

print(
    results_df.to_string(index=False)
)

# ---------------------------------
# 8. Best F1 threshold
# ---------------------------------
best_row = results_df.loc[
    results_df["F1_Score"].idxmax()
]

print("\n================================")
print("BEST BEHAVIORAL F1 THRESHOLD")
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