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
The project **Detecting Fraudulent Healthcare Providers Using Machine Learning** aims to leverage machine learning techniques to identify fraudulent behavior within Medicare claims data. Fraudulent activities such as exaggerated billing or misrepresented services are identified through patterns in claims data. The project utilizes several machine learning models including Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and LightGBM. It also applies **SMOTE (Synthetic Minority Over-sampling Technique)** to tackle the class imbalance in the dataset. A **Streamlit dashboard** was developed to offer real-time fraud detection and provide interactive visual insights to healthcare administrators. 

### Key Findings:
- **LightGBM** achieved 98% accuracy, demonstrating its strong ability to detect fraudulent providers with minimal bias.
- Important features such as **ProviderFraudRate** and **ClaimsPerProvider** were identified as key indicators of fraud.
- **SMOTE** significantly enhanced model performance by addressing class imbalance in the dataset.
- The **Streamlit dashboard** provided an intuitive interface for real-time fraud prediction and decision-making.

---

## Introduction

Healthcare fraud, especially within Medicare, leads to significant financial losses and affects the quality of care. Traditional fraud detection methods are ineffective due to the high volume and complexity of claims data. This project applies **machine learning algorithms** to automate fraud detection, offering a scalable and efficient solution. An interactive **Streamlit dashboard** was built to visualize fraud patterns and allow real-time predictions, enabling healthcare administrators to make informed decisions swiftly.

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
3. **Address the class imbalance** in the dataset using SMOTE to improve model performance.

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

- **Dropping Columns**: Columns with more than 50% missing data were removed.
- **Merging Datasets**: Merged inpatient, outpatient, and beneficiary data using **outer joins** and combined with fraud labels.
- **Null Handling**: Missing values were imputed using "unknown" for categorical columns and median/mode for numerical columns.
- **Encoding**: Categorical features were label-encoded to prepare for machine learning.

---

## Challenges

- **Class Imbalance**: Fraudulent claims are a minority class, leading to biased models.
- **Missing Data**: Handling null and missing values across multiple datasets.
- **Complexity of Claims**: Fraudulent patterns are subtle and complex, making them difficult to detect.

---

## Methodology

### Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution of features and identify important patterns. Visualization techniques were used to explore class distribution and correlations between features.

### Feature Engineering
Key features such as **ProviderFraudRate** and **ClaimsPerProvider** were engineered based on domain knowledge and the dataset's characteristics.

### Modeling and Analysis
Several machine learning models were trained, including Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and LightGBM. **LightGBM** was chosen for its excellent performance, especially in handling imbalanced data.

---

## Models Comparison

- **LightGBM** demonstrated the highest performance across multiple metrics:
  - **Recall**: 97.26%
  - **F1-Score**: 0.9832
  - **AUC**: 0.9981
- LightGBM is particularly effective for fraud detection due to its ability to handle class imbalance and its strong generalization performance.

---

## Streamlit Dashboard

### Key Features
- **Real-time fraud predictions**: Users can input data to get immediate predictions on whether a healthcare provider is fraudulent.
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


