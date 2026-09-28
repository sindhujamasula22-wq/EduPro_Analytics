import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
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

train_path = BASE_DIR / "data" / "processed" / "train_features.csv"
test_path = BASE_DIR / "data" / "processed" / "test_features.csv"

# ---------------------------------
# 2. Load data
# ---------------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

# ---------------------------------
# 3. Separate features and target
# ---------------------------------
target = "Target_Next_Course_Type"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

print("\nNumber of features:", X_train.shape[1])

# ---------------------------------
# 4. Balanced Random Forest
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
# 5. Predictions
# ---------------------------------
y_pred = model.predict(X_test)

# ---------------------------------
# 6. Evaluation
# ---------------------------------
accuracy = accuracy_score(y_test, y_pred)

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

cm = confusion_matrix(y_test, y_pred)

print("\n================================")
print("BALANCED RANDOM FOREST RESULTS")
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
    .rename(index={
        0: "Free",
        1: "Paid"
    })
)

# ---------------------------------
# 8. Feature importance
# ---------------------------------
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 10 Important Features:")

print(
    feature_importance.head(10)
    .to_string(index=False)
)