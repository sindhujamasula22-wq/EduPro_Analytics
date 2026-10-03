@"
# EduPro Learner Purchase Behavior Prediction & Education Analytics System

An end-to-end Machine Learning and Streamlit analytics system developed during the **Unified Mentor Machine Learning Internship**.

The project analyzes learner transaction behavior and uses historical learning activity to predict whether a learner's **next course transaction is likely to be Paid or Free**.

The project also provides interactive education analytics through a Streamlit dashboard.

---

## 🚀 Live Application

**Streamlit Dashboard:**  
https://eduproanalytics.streamlit.app/

**GitHub Repository:**  
https://github.com/sindhujamasula22-wq/EduPro_Analytics

---

## 📌 Project Objective

Online learning platforms generate large amounts of learner transaction data. Understanding how learners behave across multiple transactions can help identify patterns in course purchasing behavior.

The main objective of this project is to:

- Analyze learner purchasing behavior
- Study course and category distributions
- Engineer historical learner behavior features
- Predict the next course transaction type
- Compare multiple Machine Learning approaches
- Provide an interactive analytics and prediction dashboard

### Prediction Target

The model predicts:

- **0 → Free**
- **1 → Paid**

The prediction is based only on information available from the learner's **previous learning activity**, rather than information about the future transaction.

---

## 📊 Dataset Overview

The integrated EduPro dataset contains:

| Attribute | Value |
|---|---:|
| Learners | 3,000 |
| Transactions | 10,000 |
| Courses | 60 |
| Course Categories | 12 |
| Free Transactions | 6,403 |
| Paid Transactions | 3,597 |

The project uses chronological learner activity to construct a sequential Machine Learning dataset.

Because the original integrated dataset contains learner and teacher name fields, the raw dataset is **not included in the public GitHub repository**.

---

## 🔄 Machine Learning Pipeline

```text
EduPro Dataset
      ↓
Data Cleaning
      ↓
Chronological Transaction Ordering
      ↓
Learner History Feature Engineering
      ↓
Create Next Course Type Target
      ↓
Chronological Train/Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Prediction System
      ↓
Streamlit Dashboard