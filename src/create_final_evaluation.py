import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Final evaluation results from experiments
results = [
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.6400,
        "Paid_Precision": 0.0000,
        "Paid_Recall": 0.0000,
        "Paid_F1": 0.0000
    },
    {
        "Model": "Decision Tree",
        "Accuracy": 0.6329,
        "Paid_Precision": 0.4138,
        "Paid_Recall": 0.0476,
        "Paid_F1": 0.0854
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.6400,
        "Paid_Precision": 0.0000,
        "Paid_Recall": 0.0000,
        "Paid_F1": 0.0000
    },
    {
        "Model": "Balanced Random Forest",
        "Accuracy": 0.6150,
        "Paid_Precision": 0.3871,
        "Paid_Recall": 0.1190,
        "Paid_F1": 0.1821
    },
    {
        "Model": "Behavioral Baseline (0.50)",
        "Accuracy": 0.5564,
        "Paid_Precision": 0.3504,
        "Paid_Recall": 0.2718,
        "Paid_F1": 0.3061
    },
    {
        "Model": "Behavioral Baseline (0.20)",
        "Accuracy": 0.4186,
        "Paid_Precision": 0.3623,
        "Paid_Recall": 0.8095,
        "Paid_F1": 0.5006
    }
]

results_df = pd.DataFrame(results)

output_dir = (
    BASE_DIR
    / "data"
    / "processed"
)

output_path = (
    output_dir
    / "final_model_evaluation.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n================================")
print("FINAL MODEL EVALUATION")
print("================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved to:")
print(output_path)