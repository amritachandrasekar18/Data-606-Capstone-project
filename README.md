# Detecting Fraudulent Healthcare Providers Using Machine Learning

## Overview
This project aims to develop a machine learning-based solution for detecting fraudulent healthcare providers using Medicare claims data. The project leverages advanced data analysis techniques, including **SMOTE** for addressing class imbalance, and various machine learning models to identify fraudulent behaviors. The model predictions are presented in an interactive **Streamlit dashboard** that provides real-time insights and decision-making support for healthcare administrators.

## Problem Statement
In the U.S. healthcare system, fraud costs taxpayers billions of dollars every year. A real-world example is the case of a healthcare provider who submits fraudulent Medicare claims for services that were never provided, such as billing for unnecessary medical tests or overcharging for treatments. In 2019, it was reported that healthcare fraud led to losses of over $60 billion in the U.S. alone. This fraudulent behavior not only strains the financial stability of healthcare systems but also puts patient safety at risk, as resources are diverted away from legitimate care.
Traditional fraud detection methods, which often rely on manual audits and rule-based checks, struggle to keep up with the sheer volume and complexity of claims data, making it difficult to identify subtle patterns of fraud. This project aims to harness machine learning to automate the process, providing a more efficient, scalable, and accurate solution for detecting fraudulent healthcare providers, ultimately reducing financial losses and ensuring the quality and integrity of care provided to patients.

## Research Questions
1. **How effectively can machine learning algorithms identify potentially fraudulent Medicare providers based on claims data?**
2. **Which features in the dataset are most indicative of fraudulent behavior?**
3. **How can we address the class imbalance in the dataset, where fraudulent claims are far fewer than legitimate ones?**

## Background and Context
Healthcare fraud results in billions of dollars in losses each year. Providers may submit false claims for non-existent services, overstate charges, or alter billing codes. Traditional fraud detection relies on manual audits, which are time-consuming and ineffective in catching subtle fraud patterns. This project aims to automate fraud detection by applying **supervised machine learning models**, which can identify hidden patterns in large-scale claims data and improve the efficiency of fraud detection processes.

## Objectives
- **Detect fraudulent healthcare providers** using machine learning techniques.
- **Build an interactive Streamlit dashboard** for real-time fraud predictions and data visualizations.
- **Handle class imbalance** in the dataset using **SMOTE** to improve model accuracy.
- **Evaluate various machine learning models** to determine the most effective approach for fraud detection.

## Data Collection and Description
The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data), which includes **Medicare claims data** containing inpatient, outpatient, and beneficiary details. It also contains labeled examples of fraud and non-fraudulent claims.

### Dataset Components:
- **Inpatient Data**: Claims filed for hospitalized patients, including admission and discharge dates, diagnosis codes, and procedure codes.
- **Outpatient Data**: Claims filed for outpatient services.
- **Beneficiary Data**: Includes beneficiary KYC details, health conditions, and regional information.
- **Fraud Labels**: Labeled data indicating whether a provider is flagged as fraudulent or non-fraudulent.

## Methodology
### Data Preprocessing
- **Merging Datasets**: Merged inpatient, outpatient, and beneficiary datasets using outer joins, followed by an inner join with fraud labels.
- **Handling Missing Values**: Filled missing values for specific columns using appropriate methods (e.g., "unknown" for categorical features, median for numerical values).
- **Feature Engineering**: Created additional features such as **ProviderFraudRate**, **ClaimsPerProvider**, and others relevant to fraud detection.

### Model Development
- **SMOTE** was applied to handle class imbalance, improving model performance.
- Multiple machine learning models were tested, including **Logistic Regression**, **Random Forest**, **XGBoost**, **Gradient Boosting**, and **LightGBM**.
- **LightGBM** was selected as the best-performing model due to its high accuracy and ability to handle imbalanced data.

### Streamlit Dashboard
- The **Streamlit dashboard** was developed to provide real-time predictions, allowing users to interact with the model and visualize fraud patterns and features.

## Key Findings
- **LightGBM** achieved an **accuracy of 98%**, with high **precision** and **recall**.
- **ProviderFraudRate** and **ClaimsPerProvider** were identified as the most significant features for detecting fraud.
- **SMOTE** significantly improved model performance by addressing the class imbalance issue.

## Results and Insights
- The **LightGBM model** outperformed other models, demonstrating its effectiveness in fraud detection.
- The **Streamlit dashboard** allowed users to interact with the data, view fraud distributions, and make real-time predictions, providing valuable insights for healthcare administrators.

## Conclusion
This project successfully demonstrated the potential of machine learning for automating the detection of healthcare fraud. By leveraging **LightGBM** and **Streamlit**, the solution offers a scalable and efficient approach to identifying fraudulent healthcare providers in real-time. The model’s high accuracy and interactive dashboard provide a valuable tool for healthcare administrators to improve decision-making and prevent fraud.

## Future Scope
- **Anomaly Detection**: Exploring unsupervised learning techniques to detect emerging fraud patterns.
- **Real-time Integration**: Integrating the model with live healthcare claim systems for immediate fraud detection.


