import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
# 2. Load sequential dataset
# ---------------------------------
df = pd.read_csv(sequential_path)

print("Dataset shape:", df.shape)

# ---------------------------------
# 3. Sort chronologically by learner
# ---------------------------------
df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"]
)

df = df.sort_values(
    ["UserID", "TransactionDate"]
).reset_index(drop=True)

# ---------------------------------
# 4. Create behavioral prediction
# ---------------------------------
# If the learner previously purchased
# more Paid courses than Free courses,
# predict that the next course will be Paid.

df["Previous_Paid_Ratio"] = (
    df["Previous_Paid_Count"]
    / df["Previous_Transaction_Count"]
)

df["Baseline_Prediction"] = (
    df["Previous_Paid_Ratio"] >= 0.50
).astype(int)

# ---------------------------------
# 5. Chronological 80/20 split
# ---------------------------------
df = df.sort_values(
    "TransactionDate"
).reset_index(drop=True)

split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:].copy()

y_test = test_df["Target_Next_Course_Type"]
y_pred = test_df["Baseline_Prediction"]

# ---------------------------------
# 6. Evaluation
# ---------------------------------
accuracy = accuracy_score(
    y_test,
    y_pred
)

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

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n================================")
print("BEHAVIORAL BASELINE RESULTS")
print("================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Free", "Paid"],
        zero_division=0
    )
)

# ---------------------------------
# 7. Prediction distribution
# ---------------------------------
print("\nPrediction Distribution:")

print(
    pd.Series(y_pred)
    .value_counts()
    .rename(
        index={
            0: "Free",
            1: "Paid"
        }
    )
)