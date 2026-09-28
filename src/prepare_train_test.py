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
    / "ml_features.csv"
)

train_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_features.csv"
)

test_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_features.csv"
)

# ---------------------------------
# 2. Load engineered dataset
# ---------------------------------
df = pd.read_csv(input_path)

# ---------------------------------
# 3. Load original dates separately
# ---------------------------------
sequential_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_sequential_dataset.csv"
)

date_df = pd.read_csv(sequential_path)

date_df["TransactionDate"] = pd.to_datetime(
    date_df["TransactionDate"],
    errors="coerce"
)

# Make sure the row order matches
if not (
    df.index.equals(date_df.index)
):
    raise ValueError(
        "Feature dataset and date dataset "
        "do not have matching row order."
    )

# Add transaction date temporarily
df["TransactionDate"] = date_df["TransactionDate"]

# ---------------------------------
# 4. Sort chronologically
# ---------------------------------
df = df.sort_values(
    "TransactionDate"
).reset_index(drop=True)

# ---------------------------------
# 5. Time-based train/test split
# ---------------------------------
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# ---------------------------------
# 6. Separate target
# ---------------------------------
target = "Target_Next_Course_Type"

y_train = train_df[target]
y_test = test_df[target]

# Remove target and date
X_train = train_df.drop(
    columns=[
        target,
        "TransactionDate"
    ]
)

X_test = test_df.drop(
    columns=[
        target,
        "TransactionDate"
    ]
)

# ---------------------------------
# 7. Convert boolean features
# ---------------------------------
boolean_columns = X_train.select_dtypes(
    include="bool"
).columns

X_train[boolean_columns] = (
    X_train[boolean_columns].astype(float)
)

X_test[boolean_columns] = (
    X_test[boolean_columns].astype(float)
)

# ---------------------------------
# 8. Make sure train/test columns match
# ---------------------------------
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# ---------------------------------
# 9. Save train/test datasets
# ---------------------------------
train_output = X_train.copy()
train_output[target] = y_train.values

test_output = X_test.copy()
test_output[target] = y_test.values

train_output.to_csv(
    train_path,
    index=False
)

test_output.to_csv(
    test_path,
    index=False
)

# ---------------------------------
# 10. Verification
# ---------------------------------
print("\nTrain/test preparation completed!")
print("--------------------------------")

print("Total records:", len(df))
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining date range:")
print(
    train_df["TransactionDate"].min(),
    "to",
    train_df["TransactionDate"].max()
)

print("\nTesting date range:")
print(
    test_df["TransactionDate"].min(),
    "to",
    test_df["TransactionDate"].max()
)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nFeature columns:")
print(X_train.columns.tolist())

print("\nMissing values in training:")
print(X_train.isnull().sum().sum())

print("\nMissing values in testing:")
print(X_test.isnull().sum().sum())

print("\nSaved:")
print(train_path)
print(test_path)