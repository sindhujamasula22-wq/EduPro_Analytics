import pandas as pd
from pathlib import Path

# ---------------------------------
# 1. Paths
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_sequential_dataset.csv"
)

output_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_features.csv"
)

# ---------------------------------
# 2. Load sequential dataset
# ---------------------------------
df = pd.read_csv(input_path)

# Convert date
df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"],
    errors="coerce"
)

# ---------------------------------
# 3. Create date features
# ---------------------------------
df["Transaction_Year"] = (
    df["TransactionDate"].dt.year
)

df["Transaction_Month"] = (
    df["TransactionDate"].dt.month
)

df["Transaction_DayOfWeek"] = (
    df["TransactionDate"].dt.dayofweek
)

# ---------------------------------
# 4. Create learner behavior features
# ---------------------------------

df["Transaction_Frequency"] = (
    df["Previous_Transaction_Count"]
    / (df["Days_Since_First_Transaction"] + 1)
)

df["Spending_Per_Transaction"] = (
    df["Previous_Total_Spending"]
    / df["Previous_Transaction_Count"]
)

# ---------------------------------
# 5. Select ML features
# ---------------------------------
feature_columns = [
    "LearnerAge",
    "LearnerGender",

    "Previous_Transaction_Count",
    "Previous_Paid_Count",
    "Previous_Free_Count",
    "Previous_Paid_Ratio",

    "Previous_Total_Spending",
    "Previous_Average_Spending",
    "Previous_Average_Rating",
    "Previous_Average_Duration",

    "Previous_Course_Category",
    "Previous_Category_Diversity",

    "Days_Since_Previous_Transaction",
    "Days_Since_First_Transaction",

    "Transaction_Year",
    "Transaction_Month",
    "Transaction_DayOfWeek",

    "Transaction_Frequency",
    "Spending_Per_Transaction"
]

target_column = "Target_Next_Course_Type"

ml_features = df[
    feature_columns + [target_column]
].copy()

# ---------------------------------
# 6. Encode categorical features
# ---------------------------------

ml_features = pd.get_dummies(
    ml_features,
    columns=[
        "LearnerGender",
        "Previous_Course_Category"
    ],
    drop_first=True
)

# ---------------------------------
# 7. Save engineered dataset
# ---------------------------------
output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

ml_features.to_csv(
    output_path,
    index=False
)

# ---------------------------------
# 8. Verification
# ---------------------------------
print("\nFeature engineering completed!")
print("--------------------------------")

print("Original shape:", df.shape)

print(
    "ML feature shape:",
    ml_features.shape
)

print("\nFeature columns:")
print(
    ml_features.columns.tolist()
)

print("\nData types:")
print(
    ml_features.dtypes
)

print("\nMissing values:")
print(
    ml_features.isnull().sum()
)

print("\nTarget distribution:")
print(
    ml_features[target_column]
    .value_counts()
)

print("\nPrevious category features:")
print(
    [
        column
        for column in ml_features.columns
        if "Previous_Course_Category" in column
    ]
)

print("\nFirst 5 rows:")
print(
    ml_features.head()
)

print("\nSaved to:")
print(output_path)