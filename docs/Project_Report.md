\# EduPro Learner Purchase Behavior Prediction \& Education Analytics System



\## Project Report



\*\*Author:\*\* Sindhuja Masula  

\*\*Program:\*\* B.Tech Computer Science and Engineering – Artificial Intelligence and Machine Learning  

\*\*Project Type:\*\* Machine Learning Internship Project  

\*\*Organization:\*\* Unified Mentor  

\*\*Project Domain:\*\* Education Analytics / Machine Learning  



\---



\## Abstract



The EduPro Learner Purchase Behavior Prediction \& Education Analytics System is an end-to-end machine learning project designed to analyze learner transaction behavior and predict whether a learner's next course transaction is likely to be Paid or Free.



The project uses historical learner transaction data containing information about learners, courses, course categories, course types, spending, ratings, transaction dates, and other educational attributes. Instead of treating transactions as independent records, the project uses a sequential learning approach in which information from a learner's previous transactions is used to construct features for predicting the type of the learner's next course transaction.



The machine learning pipeline includes data preprocessing, chronological transaction ordering, learner-history feature engineering, sequential target creation, time-based train/test splitting, model training, evaluation, behavioral baseline comparison, and deployment through an interactive Streamlit dashboard.



Several classification approaches were evaluated, including Logistic Regression, Decision Tree, Random Forest, Balanced Random Forest, and behavioral baselines. The Balanced Random Forest was selected for the deployed prediction interface because it was able to identify a portion of Paid-course cases while maintaining a practical prediction interface.



The evaluation results also demonstrate that the available behavioral features provide limited predictive separation between future Paid and Free transactions. Therefore, the system is presented as an educational analytics and prediction-support system rather than as a highly accurate purchasing prediction engine.



\---



\## 1. Introduction



Online education platforms generate large amounts of learner interaction and transaction data. Understanding how learners behave over time can help educational platforms analyze purchasing patterns, learner engagement, course preferences, and potential transitions between Free and Paid learning experiences.



Traditional transaction-level analysis often treats every transaction independently. However, learner behavior is sequential. A learner's previous purchases, spending history, course preferences, and time between transactions may provide useful information about their future course choices.



This project investigates whether historical learner behavior can be used to predict the type of a learner's next course transaction.



The project also provides an interactive analytics dashboard through Streamlit, allowing users to explore educational transaction patterns and generate predictions based on learner history.



\---



\## 2. Problem Statement



Educational platforms may contain thousands of learner transactions across multiple courses and categories. A major analytical challenge is understanding whether previous learner behavior can provide useful information about the learner's next course purchase.



The objective of this project is therefore to develop a machine learning system that:



\- Analyzes learner transaction behavior.

\- Uses previous learner activity to construct predictive features.

\- Predicts whether the next course transaction will be Paid or Free.

\- Compares multiple machine learning approaches.

\- Evaluates the predictive limitations of the available data.

\- Provides an interactive analytics and prediction dashboard.



\---



\## 3. Objectives



The major objectives of the project are:



1\. Analyze the structure and characteristics of the EduPro learner transaction dataset.

2\. Clean and preprocess the available transaction data.

3\. Analyze learner behavior across multiple transactions.

4\. Construct sequential learner-history features.

5\. Create a target representing the type of the learner's next course transaction.

6\. Train and compare multiple machine learning classification models.

7\. Use a chronological train/test split to avoid using future information during training.

8\. Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrices.

9\. Compare machine learning predictions with a simple behavioral baseline.

10\. Deploy the analytical and prediction system through Streamlit.

11\. Clearly communicate the limitations of the predictive model.



\---



\## 4. Dataset Description



The original EduPro dataset contains:



\- \*\*10,000 transactions\*\*

\- \*\*3,000 learners\*\*

\- \*\*60 courses\*\*

\- \*\*12 course categories\*\*

\- \*\*24 original columns\*\*



Important attributes include:



\- Learner age

\- Learner gender

\- Course ID

\- Course name

\- Course category

\- Course type

\- Course level

\- Course price

\- Course duration

\- Course rating

\- Transaction date

\- Transaction amount

\- Payment method

\- Teacher information



The original dataset contains both learner-related and teacher-related attributes.



Personally identifying fields such as learner names and teacher names were not used in the public GitHub repository.



\---



\## 5. Dataset Exploration



Initial exploration identified several important characteristics of the dataset.



\### Course Type Distribution



The original transactions contain:



\- \*\*6,403 Free-course transactions\*\*

\- \*\*3,597 Paid-course transactions\*\*



Therefore, Free transactions form the larger class in the original dataset.



\### Learner Activity



The dataset contains repeated transactions for many learners.



Analysis showed:



\- \*\*1,380 learners\*\* have two or more transactions.

\- \*\*1,378 learners\*\* have transactions occurring on multiple dates.

\- The average span between a learner's first and last transaction is approximately \*\*199 days\*\*.

\- The median span is approximately \*\*207 days\*\*.

\- The maximum observed span is approximately \*\*363 days\*\*.

\- \*\*964 learners\*\* changed course type at least once.



These observations support the use of a sequential learner-history approach.



\---



\## 6. Data Preprocessing



The transaction data was processed before machine learning.



The preprocessing steps included:



1\. Converting transaction dates into proper datetime values.

2\. Converting transaction amounts into numeric values.

3\. Sorting transactions chronologically for each learner.

4\. Identifying learners with previous transaction history.

5\. Constructing historical learner features.

6\. Creating a target representing the next transaction's course type.

7\. Checking missing values and data consistency.

8\. Removing records that could not provide sufficient previous-history information.



One transaction had a missing `TransactionDate`. This record could not participate in chronological sequential modeling.



\---



\## 7. Sequential Prediction Approach



The central idea of the project is to predict a learner's next course type using information from their previous activity.



Instead of directly predicting the course type of the current transaction from information belonging to the same transaction, the project creates a historical snapshot.



For example:



\*\*Previous learner transactions\*\*



↓



\*\*Historical learner features\*\*



↓



\*\*Next transaction\*\*



↓



\*\*Target: Paid or Free\*\*



This approach helps prevent information from the future transaction from being used as an input feature.



\---



\## 8. Machine Learning Dataset



After sequential processing, the project produced:



\- \*\*6,999 usable sequential prediction records\*\*

\- \*\*18 columns before additional feature engineering\*\*



The target distribution was:



| Target | Transactions | Percentage |

|---|---:|---:|

| Free | 4,432 | 63.32% |

| Paid | 2,567 | 36.68% |



The target variable is:



\- `0` = Free

\- `1` = Paid



\---



\## 9. Feature Engineering



Historical learner behavior was converted into machine learning features.



Important features include:



\- Learner age

\- Previous transaction count

\- Previous paid count

\- Previous free count

\- Previous paid ratio

\- Previous total spending

\- Previous average spending

\- Previous average course rating

\- Previous average course duration

\- Previous course category

\- Previous category diversity

\- Days since previous transaction

\- Days since first transaction

\- Transaction year

\- Transaction month

\- Transaction day of week

\- Transaction frequency

\- Spending per transaction

\- Learner gender

\- Previous course category



Two additional behavioral features were calculated.



\### Transaction Frequency



Transaction frequency was calculated using previous transaction count and the time since the learner's first transaction:



```text

Transaction Frequency =

Previous Transaction Count /

(Days Since First Transaction + 1)

