import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_sequential_dataset.csv"
)

df = pd.read_csv(data_path)

df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"]
)

# Calculate historical paid ratio
df["Previous_Paid_Ratio"] = (
    df["Previous_Paid_Count"]
    / df["Previous_Transaction_Count"]
)

# Chronological ordering
df = df.sort_values(
    "TransactionDate"
).reset_index(drop=True)

# Test period
split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:].copy()

# Create ratio groups
test_df["Paid_Ratio_Group"] = pd.cut(
    test_df["Previous_Paid_Ratio"],
    bins=[-0.01, 0.20, 0.40, 0.60, 0.80, 1.00],
    labels=[
        "0-20%",
        "20-40%",
        "40-60%",
        "60-80%",
        "80-100%"
    ]
)

# Calculate actual Paid rate in each group
group_analysis = (
    test_df
    .groupby(
        "Paid_Ratio_Group",
        observed=False
    )
    .agg(
        Transactions=("Target_Next_Course_Type", "count"),
        Actual_Paid_Rate=("Target_Next_Course_Type", "mean"),
        Average_Previous_Paid_Ratio=(
            "Previous_Paid_Ratio",
            "mean"
        )
    )
    .reset_index()
)

group_analysis["Actual_Paid_Rate"] *= 100
group_analysis["Average_Previous_Paid_Ratio"] *= 100

print("\n================================")
print("BEHAVIORAL PERFORMANCE ANALYSIS")
print("================================")

print(
    group_analysis.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n================================")
print("OVERALL TEST PERIOD")
print("================================")

print(
    f"Transactions : {len(test_df)}"
)

print(
    f"Actual Paid  : "
    f"{test_df['Target_Next_Course_Type'].mean() * 100:.2f}%"
)

print(
    f"Actual Free  : "
    f"{(1 - test_df['Target_Next_Course_Type'].mean()) * 100:.2f}%"
)