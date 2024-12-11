# Detecting Fraudulent Healthcare Providers Using Machine Learning

**Capstone Project in Data Science - DATA 606**  
Professor: Zeynep Kacar  
By: **Amrita Chandrasekar (NH53017)**  
Date: 12/10/2024  

## Table of Contents
- [Executive Summary](#executive-summary)
- [Introduction](#introduction)
- [Research Questions](#research-questions)
- [Background and Context](#background-and-context)
- [Objectives](#objectives)
- [Data Collection and Description](#data-collection-and-description)
- [Data Cleaning](#data-cleaning)
- [Challenges](#challenges)
- [Methodology](#methodology)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Modeling and Analysis](#modeling-and-analysis)
  - [Models Comparison](#models-comparison)
- [Streamlit Dashboard](#streamlit-dashboard)
  - [Key Features](#key-features)
  - [Building the Dashboard](#building-the-dashboard)
  - [Deployment](#deployment)
- [Discussion](#discussion)
- [Conclusions and Recommendations](#conclusions-and-recommendations)
- [Future Work](#future-work)
- [References](#references)
- [Appendices](#appendices)

---

## Executive Summary
The project Detecting Fraudulent Healthcare Providers Using Machine Learning seeks to address the pervasive issue of fraudulent activities within Medicare claims data using advanced machine learning techniques. Healthcare fraud, including practices like inflated billing and falsified services, is a major concern that leads to financial losses and undermines the integrity of the healthcare system. By applying various machine learning models such as Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and LightGBM, the project identifies patterns indicative of fraud in the claims data. Additionally, the class imbalance in the dataset is mitigated using SMOTE (Synthetic Minority Over-sampling Technique), which enhances the model’s ability to accurately detect fraudulent claims. The project also features an interactive Streamlit dashboard, which allows healthcare administrators to conduct visualize key insights, predict fraud/non-fraud facilitating more informed decision-making and improving overall fraud prevention efforts.

### Key Findings:
- **LightGBM** achieved 98% accuracy, demonstrating its strong ability to detect fraudulent providers with minimal bias.
- Important features such as **ProviderFraudRate** and **ClaimsPerProvider** were identified as key indicators of fraud.
- **SMOTE** significantly enhanced model performance by addressing class imbalance in the dataset.
- The **Streamlit dashboard** provided an intuitive interface for real-time fraud prediction and decision-making.

---

## Introduction

Healthcare fraud, especially within Medicare, leads to significant financial losses and affects the quality of care. Traditional fraud detection methods are ineffective due to the high volume and complexity of claims data. This project applies **machine learning algorithms** to automate fraud detection, offering a scalable and efficient solution. An interactive **Streamlit dashboard** was built to visualize fraud patterns and allow real-time predictions, enabling healthcare administrators to make informed decisions swiftly.

---

## Problem Statement
In the U.S. healthcare system, fraud costs taxpayers billions of dollars every year. A real-world example is the case of a healthcare provider who submits fraudulent Medicare claims for services that were never provided, such as billing for unnecessary medical tests or overcharging for treatments. In 2019, it was reported that healthcare fraud led to losses of over $60 billion in the U.S. alone. This fraudulent behavior not only strains the financial stability of healthcare systems but also puts patient safety at risk, as resources are diverted away from legitimate care.

Traditional fraud detection methods, which often rely on manual audits and rule-based checks, struggle to keep up with the sheer volume and complexity of claims data, making it difficult to identify subtle patterns of fraud. This project aims to harness machine learning to automate the process, providing a more efficient, scalable, and accurate solution for detecting fraudulent healthcare providers, ultimately reducing financial losses and ensuring the quality and integrity of care provided to patients.

---

## Research Questions

- **How effectively can machine learning algorithms identify potentially fraudulent Medicare providers based on claims data?**
- **Which features in the dataset are most indicative of fraudulent behavior?**
- **How can we address the class imbalance in the dataset, where fraudulent claims are far fewer than legitimate ones?**

---

## Background and Context

Healthcare fraud is a pervasive problem that drains financial resources and jeopardizes patient care. The challenge lies in identifying subtle fraud patterns in massive datasets. By applying machine learning, particularly **supervised learning models**, this project aims to create a system that can efficiently predict fraudulent claims. The dataset used includes **inpatient**, **outpatient**, and **beneficiary** data, each containing features related to healthcare providers, services rendered, and claims.

---

## Objectives

1. **Detect fraudulent healthcare providers** using machine learning algorithms.
2. **Build an interactive Streamlit dashboard** for real-time predictions and data visualizations.

---

## Data Collection and Description

The dataset was sourced from **Kaggle** and includes a range of claims data:

- **Inpatient Data**: Claims for patients admitted to hospitals, including admission/discharge dates and diagnosis codes.
- **Outpatient Data**: Claims for patients who received outpatient services.
- **Beneficiary Data**: Demographic and health condition information for each Medicare beneficiary.
- **Fraud Labels**: Labeled data indicating whether a provider is fraudulent or not.

Dataset link: [Healthcare Provider Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data)

---

## Data Cleaning

- Dropping columns that have more than 50% null values for inpatient, outpatient and beneficiary datasets individually.
- Merging “inpatient” and “outpatient” as “merged_data” with the help of outer join function. Then merging “merged_data” with beneficiary dataset on BeneID column with outer join function as “final merged” dataset.
- Shape of Final Merged Dataset: (558211, 46)
- Now combining “final merged” dataset with “train dataset” on provider column with the help of inner join as “df_final”
- Shape of df_final dataset: (558211, 47).
- Removing columns from df_final that have more than 50%null values as they are not part of analysis.
- Shape of df_final dataset:(558211, 34)
- Checking for duplicate values
- Checking for null values: AttendingPhysician 1508, DeductibleAmtPaid 899, ClmDiagnosisCode_1 10453, ClmDiagnosisCode_2 195606
- To handle null values, “unknown” is used in place of null values for the column “AttendingPhysician” while “median” and “mode” is used for columns “DeductibleAmtPaid”,”ClmDiagnosisCode_1” and “ClmDiagnosisCode_2” respectively.
- Converting date to datetime format
- Performed label encoding to convert categorical features to numerical features for making data ready for visualizations![image](https://github.com/user-attachments/assets/10308d2b-7b4c-4fd1-bcea-295b09a1080a)


---

## Challenges

- **Class Imbalance**: Fraudulent claims are a minority class, leading to biased models.
- <img width="274" alt="image" src="https://github.com/user-attachments/assets/ba7e6c0a-d05b-4314-9cdd-cc3a1af1dc35">
- <img width="206" alt="image" src="https://github.com/user-attachments/assets/c6fb50fe-1dbc-4f94-8a27-cf7928c99f6f">



---

## Methodology

### Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution of features and identify important patterns. Visualization techniques were used to explore class distribution and correlations between features.

<img width="396" alt="image" src="https://github.com/user-attachments/assets/e3743aad-69e8-4ac8-94f2-ecab47a6a223">
<img width="396" alt="image" src="https://github.com/user-attachments/assets/51634151-e4b7-4243-91e9-c9c1d4620a9c">
<img width="357" alt="image" src="https://github.com/user-attachments/assets/983764b7-81a1-4107-a4f6-c0b7bf0a83a4">
<img width="375" alt="image" src="https://github.com/user-attachments/assets/9743773f-904f-44d1-81a6-4445b5dfc51a">
<img width="394" alt="image" src="https://github.com/user-attachments/assets/b2d027c9-b623-49b9-9885-59d593ed8d3b">
<img width="298" alt="image" src="https://github.com/user-attachments/assets/30402ebb-1284-487f-9e65-edb65445106b">





### Feature Engineering
Key features such as **ProviderFraudRate** and **ClaimsPerProvider** were engineered based on domain knowledge and the dataset's characteristics.

<img width="436" alt="image" src="https://github.com/user-attachments/assets/3022003a-3fc2-4f6c-b48c-a88312715212">

- CostPerDay: This feature is moderately correlated with potential fraud (0.11) and can help identify unusually high costs per day for providers, which could indicate fraud.
- DeductibleRatio: With a very low correlation to potential fraud, this feature may be less impactful in identifying fraud but could provide additional context on healthcare cost structures.
- ChronicConditionCount: This feature has a weak negative correlation with potential fraud, suggesting it might not strongly indicate fraudulent behavior but could be useful in understanding patient risk.
- GenderCostRatio: Shows a weak positive correlation with potential fraud (0.09), which might be useful for detecting anomalies in cost allocation across genders.
- ClaimsPerProvider: This feature strongly correlates with potential fraud (0.33), as a higher number of claims per provider could signal fraudulent activity.
- AvgProviderReimbursement: Correlated at 0.21 with potential fraud, this feature is relevant for detecting discrepancies between expected and actual reimbursements, indicating potential fraud.
- ProviderFraudRate: This column is highly correlated with potential fraud (0.85), making it one of the most critical features for detecting fraudulent behavior in healthcare providers.
- CostPerCoverageMonth: Strong correlation with potential fraud (0.74), suggesting that unusually high costs per coverage month could be a strong indicator of fraudulent activity.
- DurationReimbursement: With a weaker correlation to potential fraud (0.05), this feature might not be highly indicative of fraud but could still be valuable in a broader fraud detection model.
- StateFraudRate: While not highly correlated with potential fraud, this feature (0.28) provides geographic context, potentially useful for understanding state-level fraud trends.
- PotentialFraud: This is the target variable in your analysis, with a correlation of 1, making it central to the project for identifying fraudulent healthcare providers.

<img width="356" alt="image" src="https://github.com/user-attachments/assets/b28d1cd4-fd73-4c51-833d-b946d6cc7c41">

### Feature Importance Analysis
The process begins by preparing the data, where categorical variables are one-hot encoded, and the dataset is split into training and testing sets. The presence of infinite and NaN values is checked and handled by replacing infinite values with the column maximum. A Random Forest model is then trained on the data, and feature importance is computed to identify the top features most strongly correlated with the target variable "PotentialFraud.The top features are displayed to guide model implementation. This analysis is essential for further model development and integration into a Streamlit dashboard for interactive fraud detection.

<img width="356" alt="image" src="https://github.com/user-attachments/assets/12926589-f2fc-4788-98b8-9c24eedd67b0">




### Modeling and Analysis
Several machine learning models were trained, including Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and LightGBM. **LightGBM** was chosen for its excellent performance, especially in handling imbalanced data.
### Logistic Regression
<img width="188" alt="image" src="https://github.com/user-attachments/assets/81f4973d-231d-4d93-9236-c755c6ea3f9a">

### Random Forest
<img width="183" alt="image" src="https://github.com/user-attachments/assets/29acaefe-f2da-4067-b426-2c8cf2aa78d1">

### XGBoost
<img width="170" alt="image" src="https://github.com/user-attachments/assets/ad07376a-b9c7-492d-bd82-8e1455875add">

### Lightgbm
<img width="196" alt="image" src="https://github.com/user-attachments/assets/df1861ad-0335-4a35-bb72-219ceca08972">

### gradient boosting
<img width="194" alt="image" src="https://github.com/user-attachments/assets/0c489e76-5676-4a43-bde3-f2528391a807">


## Models Comparison
<img width="400" alt="image" src="https://github.com/user-attachments/assets/d80d0312-2bfe-488e-82af-4fc046e68c2f">

<img width="286" alt="image" src="https://github.com/user-attachments/assets/846bea8a-8765-4640-843d-0dcfdc23ba06">


- **LightGBM** demonstrated the highest performance across multiple metrics:
  - **Recall**: 97.26%
  - **F1-Score**: 0.9832
  - **AUC**: 0.9981
- LightGBM is particularly effective for fraud detection due to its ability to handle class imbalance and its strong generalization performance.



## Streamlit Dashboard

### Key Features
- **Fraud predictions**: Users can input data to get immediate predictions on whether a healthcare provider is fraudulent.
- **Visualizations**:
  - **Fraud vs. Non-Fraud distribution**
  - **State-wise fraud rates**
  - **Correlation heatmap** for key features
- **User-Friendly Interface**: The dashboard is intuitive and requires no coding experience.

### Building the Dashboard
- **Data Integration**: Data was preprocessed, merged, and ready for integration into Streamlit.
- **Model Integration**: The trained LightGBM model was incorporated using **joblib** for predictions.

### Deployment
The Streamlit app was deployed on the **Streamlit Community Cloud**, making it easily accessible via a web link.

---

## Discussion

### Interpretation of Results
- The LightGBM model’s high performance (98% accuracy) and the use of **SMOTE** for class balancing were pivotal in achieving strong results.
- The **Streamlit dashboard** enhanced the model's usability, providing healthcare professionals with quick insights.

### Comparison with Existing Literature
- The findings align with other studies on healthcare fraud detection, which emphasize the importance of machine learning and handling class imbalance.

### Unexpected Findings
- The significance of **ProviderFraudRate** in detecting fraud was unexpected and highlights the value of feature engineering.

---

## Conclusions and Recommendations

### Key Findings
- **LightGBM** achieved an accuracy of 98%, with **ProviderFraudRate** being the most critical feature.
- SMOTE improved performance by addressing class imbalance.

### Recommendations
- Implement continuous fraud monitoring using the LightGBM model.
- Expand features with additional data sources for better fraud detection.

---

## Future Work

1. **Anomaly Detection**: Incorporate unsupervised models to detect unknown fraud patterns.
2. **Real-time Integration**: Integrate the model with live claim systems for faster fraud detection.

---

## References
1. **Bauder, R.A., & Khoshgoftaar, T.M. (2017)**. *Medicare fraud detection using machine learning methods.* IEEE ICMLA.
2. **Garmdareh, M.S., et al. (2023)**. *A Machine Learning-based Approach for Medical Insurance Anomaly Detection by Predicting Indirect Outpatients' Claim Price.* IEEE ICWR.

---
## Project links

### Data Merging, Cleaning, EDA, and ML Models
- [Google Colab Notebook: Data Merging](https://colab.research.google.com/drive/1FB09GuFramK_HQWfzsPQPdcUB4zEtXGN?usp=sharing)

### Additional Colab Notebooks
- [Notebook 2: Data Cleaning](https://colab.research.google.com/drive/1-Nd6d3xuog2EPYfJ3EoWyfh3mE_6RTMD?usp=chrome_ntp)
- [Notebook 3: Data Visualizations](https://colab.research.google.com/drive/1Flk6lF3OfQR4pJCm_cn3AVsGVold49TF?usp=chrome_ntp)
- [Notebook 4: Machine Learning Model ](https://colab.research.google.com/drive/1hmXbl07e4LIiAs2qVi24uVYqtMmaCGY7?usp=chrome_ntp)

### Streamlit App
- [Interactive Streamlit App for Visualization](https://amritachandrasekar18-data-690-capstone-project-app-pwbhji.streamlit.app/)

