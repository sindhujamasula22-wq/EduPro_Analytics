import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="EduPro Analytics",
    page_icon="🎓",
    layout="wide"
)


# =========================================
# PATHS
# =========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "original"
    / "edupro_integrated.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "balanced_random_forest.pkl"
)


# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv(DATA_PATH)

df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"],
    errors="coerce"
)

df["Amount"] = pd.to_numeric(
    df["Amount"],
    errors="coerce"
)


# =========================================
# LOAD MODEL
# =========================================

model = joblib.load(MODEL_PATH)


# =========================================
# BASIC DATASET METRICS
# =========================================

total_learners = df["UserID"].nunique()
total_transactions = len(df)
total_courses = df["CourseID"].nunique()
total_categories = df["CourseCategory"].nunique()


# =========================================
# TITLE
# =========================================

st.title("🎓 EduPro Learner Analytics")

st.subheader(
    "Learner Purchase Behavior Prediction & Education Analytics System"
)

st.markdown(
    """
    This dashboard analyzes learner transaction behavior and provides
    data-driven insights into future course purchase behavior.
    """
)

st.divider()


# =========================================
# TOP METRICS
# =========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Learners",
        f"{total_learners:,}"
    )

with col2:
    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with col3:
    st.metric(
        "Courses",
        f"{total_courses:,}"
    )

with col4:
    st.metric(
        "Course Categories",
        f"{total_categories:,}"
    )


st.divider()


# =========================================
# OVERVIEW ANALYTICS
# =========================================

st.header("📈 Overview Analytics")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Course Type Distribution")

    course_type_counts = (
        df["CourseType"]
        .value_counts()
        .rename_axis("Course Type")
        .to_frame("Transactions")
    )

    st.bar_chart(course_type_counts)


with col2:

    st.subheader("Course Category Distribution")

    category_counts = (
        df["CourseCategory"]
        .value_counts()
        .rename_axis("Course Category")
        .to_frame("Transactions")
    )

    st.bar_chart(category_counts)


st.divider()


# =========================================
# LEARNER BEHAVIOR ANALYTICS
# =========================================

st.header("👥 Learner Behavior Analytics")


learner_summary = (
    df.groupby("UserID")
    .agg(
        Transactions=("TransactionID", "count"),
        Total_Spending=("Amount", "sum")
    )
    .reset_index()
)


col1, col2, col3 = st.columns(3)


with col1:

    avg_transactions = learner_summary["Transactions"].mean()

    st.metric(
        "Average Transactions per Learner",
        f"{avg_transactions:.2f}"
    )


with col2:

    avg_spending = learner_summary["Total_Spending"].mean()

    st.metric(
        "Average Spending per Learner",
        f"{avg_spending:.2f}"
    )


with col3:

    paid_transactions = (
        df["CourseType"] == "Paid"
    ).sum()

    paid_percentage = (
        paid_transactions / len(df) * 100
    )

    st.metric(
        "Paid Transaction Rate",
        f"{paid_percentage:.2f}%"
    )


col1, col2 = st.columns(2)


with col1:

    st.subheader("Transactions per Learner")

    transaction_distribution = (
        learner_summary["Transactions"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(transaction_distribution)


with col2:

    st.subheader("Learner Spending Distribution")

    spending_distribution = (
        learner_summary["Total_Spending"]
        .round(-1)
        .value_counts()
        .sort_index()
    )

    st.line_chart(spending_distribution)


st.divider()


# =========================================
# PURCHASE PREDICTION
# =========================================

st.header("🔮 Purchase Prediction")

st.markdown(
    """
    Enter a learner's previous learning activity to estimate whether
    their **next course transaction** is more likely to be Paid or Free.
    """
)


# -----------------------------------------
# INPUT SECTION
# -----------------------------------------

st.subheader("Learner Information")

col1, col2, col3 = st.columns(3)


with col1:

    learner_age = st.number_input(
        "Learner Age",
        min_value=10,
        max_value=100,
        value=25,
        step=1
    )


with col2:

    learner_gender = st.selectbox(
        "Learner Gender",
        ["Female", "Male"]
    )


with col3:

    previous_transaction_count = st.number_input(
        "Previous Transaction Count",
        min_value=1,
        max_value=100,
        value=2,
        step=1
    )


st.subheader("Previous Learning Behavior")


col1, col2, col3 = st.columns(3)


with col1:

    previous_paid_count = st.number_input(
        "Previous Paid Courses",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    )


with col2:

    previous_free_count = st.number_input(
        "Previous Free Courses",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    )


with col3:

    previous_total_spending = st.number_input(
        "Previous Total Spending",
        min_value=0.0,
        value=100.0,
        step=10.0
    )


col1, col2, col3 = st.columns(3)


with col1:

    previous_average_spending = st.number_input(
        "Previous Average Spending",
        min_value=0.0,
        value=50.0,
        step=10.0
    )


with col2:

    previous_average_rating = st.number_input(
        "Previous Average Course Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )


with col3:

    previous_average_duration = st.number_input(
        "Previous Average Course Duration",
        min_value=0.0,
        value=30.0,
        step=1.0
    )


col1, col2, col3 = st.columns(3)


with col1:

    previous_category_diversity = st.number_input(
        "Previous Category Diversity",
        min_value=1,
        max_value=12,
        value=1,
        step=1
    )


with col2:

    days_since_previous = st.number_input(
        "Days Since Previous Transaction",
        min_value=0,
        max_value=1000,
        value=30,
        step=1
    )


with col3:

    days_since_first = st.number_input(
        "Days Since First Transaction",
        min_value=0,
        max_value=2000,
        value=100,
        step=1
    )


st.subheader("Current Transaction Timing")


col1, col2, col3 = st.columns(3)


with col1:

    transaction_year = st.number_input(
        "Transaction Year",
        min_value=2020,
        max_value=2035,
        value=2025,
        step=1
    )


with col2:

    transaction_month = st.number_input(
        "Transaction Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1
    )


with col3:

    transaction_day_of_week = st.number_input(
        "Transaction Day of Week",
        min_value=0,
        max_value=6,
        value=2,
        step=1,
        help="Monday = 0, Tuesday = 1, ..., Sunday = 6"
    )


previous_categories = [
    "Artificial Intelligence",
    "Business",
    "Cybersecurity",
    "Data Science",
    "Design",
    "Digital Marketing",
    "Finance",
    "Machine Learning",
    "Marketing",
    "Programming",
    "Project Management",
    "Web Development"
]


previous_course_category = st.selectbox(
    "Previous Course Category",
    previous_categories
)


# =========================================
# CALCULATED FEATURES
# =========================================

previous_paid_ratio = (
    previous_paid_count
    / previous_transaction_count
)


transaction_frequency = (
    previous_transaction_count
    / (days_since_first + 1)
)


spending_per_transaction = (
    previous_total_spending
    / previous_transaction_count
)


# =========================================
# PREDICTION BUTTON
# =========================================

if st.button(
    "🔮 Predict Next Course Type",
    type="primary"
):

    # -------------------------------------
    # Create all 29 model features
    # -------------------------------------

    input_data = {
        "LearnerAge": learner_age,

        "Previous_Transaction_Count":
            previous_transaction_count,

        "Previous_Paid_Count":
            previous_paid_count,

        "Previous_Free_Count":
            previous_free_count,

        "Previous_Paid_Ratio":
            previous_paid_ratio,

        "Previous_Total_Spending":
            previous_total_spending,

        "Previous_Average_Spending":
            previous_average_spending,

        "Previous_Average_Rating":
            previous_average_rating,

        "Previous_Average_Duration":
            previous_average_duration,

        "Previous_Category_Diversity":
            previous_category_diversity,

        "Days_Since_Previous_Transaction":
            days_since_previous,

        "Days_Since_First_Transaction":
            days_since_first,

        "Transaction_Year":
            transaction_year,

        "Transaction_Month":
            transaction_month,

        "Transaction_DayOfWeek":
            transaction_day_of_week,

        "Transaction_Frequency":
            transaction_frequency,

        "Spending_Per_Transaction":
            spending_per_transaction,

        "LearnerGender_Male":
            1 if learner_gender == "Male" else 0
    }


    # -------------------------------------
    # Add course category one-hot features
    # -------------------------------------

    category_features = [
        "Business",
        "Cybersecurity",
        "Data Science",
        "Design",
        "Digital Marketing",
        "Finance",
        "Machine Learning",
        "Marketing",
        "Programming",
        "Project Management",
        "Web Development"
    ]


    for category in category_features:

        column_name = (
            "Previous_Course_Category_"
            + category
        )

        input_data[column_name] = (
            1
            if previous_course_category == category
            else 0
        )


    # -------------------------------------
    # Convert to DataFrame
    # -------------------------------------

    input_df = pd.DataFrame([input_data])


    # -------------------------------------
    # Force exact model feature order
    # -------------------------------------

    input_df = input_df[
        model.feature_names_in_
    ]


    # -------------------------------------
    # Prediction
    # -------------------------------------

    prediction = model.predict(input_df)[0]

    prediction_probability = (
        model.predict_proba(input_df)[0]
    )


    # -------------------------------------
    # Display result
    # -------------------------------------

    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.success(
            "Predicted Next Course Type: **PAID** 💳"
        )

    else:

        st.info(
            "Predicted Next Course Type: **FREE** 🆓"
        )


    # -------------------------------------
    # Model score
    # -------------------------------------

    paid_probability = prediction_probability[1]
    free_probability = prediction_probability[0]


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Model Score — Paid",
            f"{paid_probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Model Score — Free",
            f"{free_probability * 100:.2f}%"
        )


    st.caption(
        "These values are model output scores from the trained "
        "Random Forest and should not be interpreted as guaranteed "
        "purchase probabilities."
    )


    # -------------------------------------
    # Show calculated historical features
    # -------------------------------------

    with st.expander(
        "View Calculated Model Features"
    ):

        display_features = pd.DataFrame(
            {
                "Feature": [
                    "Previous Paid Ratio",
                    "Transaction Frequency",
                    "Spending Per Transaction"
                ],
                "Value": [
                    previous_paid_ratio,
                    transaction_frequency,
                    spending_per_transaction
                ]
            }
        )

        st.dataframe(
            display_features,
            use_container_width=True,
            hide_index=True
        )


st.divider()


# =========================================
# MODEL EVALUATION
# =========================================

st.header("🤖 Model Evaluation")

st.markdown(
    """
    The following results compare the classification models evaluated
    during the machine-learning experiment. The evaluation uses the
    chronological test set, where the model predicts the learner's
    next course type as Paid or Free.
    """
)


# -----------------------------------------
# Load evaluation results
# -----------------------------------------

EVALUATION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_model_evaluation.csv"
)


evaluation_df = pd.read_csv(EVALUATION_PATH)


# -----------------------------------------
# Format table for display
# -----------------------------------------

evaluation_display = evaluation_df.copy()

evaluation_display["Accuracy"] = (
    evaluation_display["Accuracy"] * 100
).round(2)

evaluation_display["Paid_Precision"] = (
    evaluation_display["Paid_Precision"] * 100
).round(2)

evaluation_display["Paid_Recall"] = (
    evaluation_display["Paid_Recall"] * 100
).round(2)

evaluation_display["Paid_F1"] = (
    evaluation_display["Paid_F1"] * 100
).round(2)


evaluation_display = evaluation_display.rename(
    columns={
        "Model": "Model",
        "Accuracy": "Accuracy (%)",
        "Paid_Precision": "Paid Precision (%)",
        "Paid_Recall": "Paid Recall (%)",
        "Paid_F1": "Paid F1 (%)"
    }
)


# -----------------------------------------
# Display comparison table
# -----------------------------------------

st.subheader("Classification Model Comparison")

st.dataframe(
    evaluation_display,
    use_container_width=True,
    hide_index=True
)


# =========================================
# ACCURACY COMPARISON
# =========================================

st.subheader("Accuracy Comparison")

accuracy_chart = (
    evaluation_df
    .set_index("Model")[["Accuracy"]]
    .rename(columns={"Accuracy": "Accuracy"})
)

st.bar_chart(accuracy_chart)


# =========================================
# PAID CLASS PERFORMANCE
# =========================================

st.subheader("Paid-Class Detection Performance")

paid_metrics = (
    evaluation_df
    .set_index("Model")[
        [
            "Paid_Precision",
            "Paid_Recall",
            "Paid_F1"
        ]
    ]
)

st.bar_chart(paid_metrics)


# =========================================
# INTERPRETATION
# =========================================

st.subheader("📌 Interpretation")

st.markdown(
    """
    ### What the results show

    - The test set contains both Free and Paid next-course outcomes.
    - Accuracy alone does not fully describe model performance because
      the Free class is more common than the Paid class.
    - Some models predicted very few or no Paid transactions at their
      default classification threshold.
    - The Balanced Random Forest was able to identify a portion of the
      Paid cases while maintaining moderate overall accuracy.
    - The behavioral baseline demonstrates that simple historical
      learner behavior can also provide predictive signal.
    - The results indicate **limited predictive separation** between
      future Paid and Free transactions in the available dataset.

    Therefore, the prediction component should be treated as a
    **decision-support feature rather than a guaranteed purchase
    prediction system**.
    """
)


st.divider()


# =========================================
# BEHAVIORAL INSIGHTS
# =========================================

st.header("💡 Behavioral Insights")

st.markdown(
    """
    The analysis below examines whether previous learner behavior
    provides useful information about the next course transaction.
    """
)


# -----------------------------------------
# Paid ratio analysis
# -----------------------------------------

behavioral_groups = pd.DataFrame(
    {
        "Paid Ratio Group": [
            "0–20%",
            "20–40%",
            "40–60%",
            "60–80%",
            "80–100%"
        ],
        "Transactions": [
            288,
            550,
            413,
            62,
            87
        ],
        "Actual Paid Rate (%)": [
            34.38,
            36.55,
            36.56,
            29.03,
            40.23
        ]
    }
)


st.subheader(
    "Previous Paid Ratio vs Next Paid Transaction"
)


st.dataframe(
    behavioral_groups,
    use_container_width=True,
    hide_index=True
)


st.line_chart(
    behavioral_groups.set_index(
        "Paid Ratio Group"
    )[["Actual Paid Rate (%)"]]
)


st.markdown(
    """
    ### Key observation

    The relationship between previous Paid behavior and the next
    transaction is **not strongly monotonic** in the test period.

    This means that a learner's historical Paid ratio alone does not
    provide a consistently increasing signal for whether the next
    transaction will be Paid.
    """
)


# =========================================
# DASHBOARD MODULES
# =========================================

st.header("📊 Dashboard Modules")

st.markdown(
    """
    ### 1. 📈 Overview
    Explore overall learner and course activity.

    ### 2. 👥 Learner Behavior
    Analyze transaction frequency, spending patterns,
    previous Paid/Free behavior, and category diversity.

    ### 3. 🔮 Purchase Prediction
    Predict whether a learner's next course transaction
    is likely to be Paid or Free.

    ### 4. 🤖 Model Evaluation
    Compare the classification models evaluated during
    the machine-learning experiment.

    ### 5. 💡 Behavioral Insights
    Explore relationships between historical learner
    behavior and subsequent course purchases.
    """
)


st.divider()


# =========================================
# DISCLAIMER
# =========================================

st.info(
    "The prediction component is based on historical learner "
    "transaction behavior and should be interpreted as a "
    "decision-support tool rather than a guaranteed outcome."
)


st.caption(
    "EduPro Analytics | Machine Learning Internship Project"
)