import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
# 2. Load train and test data
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
# 4. Feature scaling
# ---------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------
# 5. Train Logistic Regression
# ---------------------------------
model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ---------------------------------
# 6. Make predictions
# ---------------------------------
y_pred = model.predict(X_test_scaled)

# ---------------------------------
# 7. Evaluation
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
print("LOGISTIC REGRESSION RESULTS")
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
# 8. Prediction distribution
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
# 9. Feature coefficients
# ---------------------------------
coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\nTop 10 Important Features:")
print(
    coefficients[
        ["Feature", "Coefficient"]
    ].head(10).to_string(index=False)
)