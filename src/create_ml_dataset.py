import pandas as pd
from pathlib import Path

# -----------------------------
# 1. Load dataset
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "original" / "edupro_integrated.csv"
output_path = BASE_DIR / "data" / "processed" / "ml_sequential_dataset.csv"

df = pd.read_csv(input_path)

# -----------------------------
# 2. Clean transaction date
# -----------------------------
df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"],
    errors="coerce"
)

df = df.dropna(subset=["TransactionDate"]).copy()

# -----------------------------
# 3. Sort chronologically
# -----------------------------
df = df.sort_values(
    ["UserID", "TransactionDate", "TransactionID"]
).reset_index(drop=True)

# -----------------------------
# 4. Create sequential records
# -----------------------------
records = []

for user_id, group in df.groupby("UserID"):

    group = group.reset_index(drop=True)

    # Need at least 2 transactions
    if len(group) < 2:
        continue

    for i in range(1, len(group)):

        current = group.iloc[i]
        history = group.iloc[:i]

        # -----------------------------
        # Previous transaction features
        # -----------------------------
        previous_transaction_count = len(history)

        previous_paid_count = (
            history["CourseType"] == "Paid"
        ).sum()

        previous_free_count = (
            history["CourseType"] == "Free"
        ).sum()

        previous_total_spending = (
            history["Amount"].sum()
        )

        previous_average_spending = (
            history["Amount"].mean()
        )

        previous_average_rating = (
            history["CourseRating"].mean()
        )

        previous_average_duration = (
            history["CourseDuration"].mean()
        )

        # -----------------------------
        # Previous course category
        # -----------------------------
        previous_course_category = (
            history["CourseCategory"].iloc[-1]
        )

        previous_category_diversity = (
            history["CourseCategory"].nunique()
        )

        # -----------------------------
        # Time-based features
        # -----------------------------
        previous_date = (
            history["TransactionDate"].iloc[-1]
        )

        first_date = (
            history["TransactionDate"].iloc[0]
        )

        days_since_previous_transaction = (
            current["TransactionDate"] - previous_date
        ).days

        days_since_first_transaction = (
            current["TransactionDate"] - first_date
        ).days

        # -----------------------------
        # Learner information
        # -----------------------------
        learner_age = current["LearnerAge"]
        learner_gender = current["LearnerGender"]

        # -----------------------------
        # Target
        # -----------------------------
        target = (
            1 if current["CourseType"] == "Paid" else 0
        )

        records.append({
            "UserID": user_id,
            "TransactionID": current["TransactionID"],
            "TransactionDate": current["TransactionDate"],

            "LearnerAge": learner_age,
            "LearnerGender": learner_gender,

            "Previous_Transaction_Count":
                previous_transaction_count,

            "Previous_Paid_Count":
                previous_paid_count,

            "Previous_Free_Count":
                previous_free_count,

            "Previous_Paid_Ratio":
                previous_paid_count / previous_transaction_count,

            "Previous_Total_Spending":
                previous_total_spending,

            "Previous_Average_Spending":
                previous_average_spending,

            "Previous_Average_Rating":
                previous_average_rating,

            "Previous_Average_Duration":
                previous_average_duration,

            "Previous_Course_Category":
                previous_course_category,

            "Previous_Category_Diversity":
                previous_category_diversity,

            "Days_Since_Previous_Transaction":
                days_since_previous_transaction,

            "Days_Since_First_Transaction":
                days_since_first_transaction,

            "Target_Next_Course_Type":
                target
        })

# -----------------------------
# 5. Create ML dataset
# -----------------------------
ml_df = pd.DataFrame(records)

# -----------------------------
# 6. Save dataset
# -----------------------------
output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

ml_df.to_csv(
    output_path,
    index=False
)

# -----------------------------
# 7. Verification
# -----------------------------
print("\nML dataset created successfully!")
print("--------------------------------")
print("Shape:", ml_df.shape)

print("\nColumns:")
print(ml_df.columns.tolist())

print("\nTarget distribution:")
print(
    ml_df["Target_Next_Course_Type"]
    .value_counts()
    .sort_index()
)

print("\nTarget percentages:")
print(
    ml_df["Target_Next_Course_Type"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nMissing values:")
print(ml_df.isnull().sum())

print("\nPrevious course category distribution:")
print(
    ml_df["Previous_Course_Category"]
    .value_counts()
)

print("\nFirst 5 rows:")
print(ml_df.head())

print("\nSaved to:")
print(output_path)