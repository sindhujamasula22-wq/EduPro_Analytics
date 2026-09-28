import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


# -----------------------------------------
# Paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_features.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "balanced_random_forest.pkl"
)


# -----------------------------------------
# Load training data
# -----------------------------------------

train_df = pd.read_csv(
    TRAIN_PATH
)

X_train = train_df.drop(
    columns=["Target_Next_Course_Type"]
)

y_train = train_df[
    "Target_Next_Course_Type"
]


# -----------------------------------------
# Train balanced Random Forest
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# -----------------------------------------
# Save model
# -----------------------------------------

joblib.dump(
    model,
    MODEL_PATH
)


print("\n================================")
print("PREDICTION MODEL SAVED")
print("================================")

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Training records: {len(X_train)}"
)

print(
    f"Features: {X_train.shape[1]}"
)

print(
    "Model type: Balanced Random Forest"
)