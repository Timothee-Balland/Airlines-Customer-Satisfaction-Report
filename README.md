# Airlines Customer Satisfaction Analysis

![GitHub last commit](https://img.shields.io/badge/last%20commit-September%202025-brightgreen)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project analyzes customer satisfaction in the airline industry using the **Airline Passenger Satisfaction** dataset. The goal is to identify the key drivers of satisfaction and build predictive models to classify passengers as Satisfied or Neutral/Dissatisfied. The dataset contains **103,904 instances** and **25 features**, including demographics, service ratings, and operational metrics.

The analysis combines **descriptive statistics** and **machine learning** techniques to provide actionable insights for airlines aiming to improve passenger experience.

---

## Features

- **Descriptive Analysis:**
  - Summary statistics for numerical and categorical variables
  - Correlation analysis and visualizations
  - Statistical tests (t-tests, chi-square) to identify key satisfaction drivers
- **Machine Learning Models:**
  - Decision Tree
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
- **Hyperparameter Optimization:** Grid Search for Decision Tree and Random Forest
- **Feature Importance Analysis:** Identifies the most influential factors for satisfaction

---

## Methodology

### 1. Descriptive Analysis
- Explored distributions of passenger demographics, travel type, class, and service ratings.
- Conducted correlation analysis to find relationships between satisfaction and features.
- Performed t-tests and chi-square tests to determine statistically significant differences between satisfied and dissatisfied passengers.

### 2. Machine Learning
- Implemented supervised models to predict passenger satisfaction:
  - **Decision Tree**: High interpretability and robust performance.
  - **Logistic Regression**: Baseline linear model for classification.
  - **Random Forest**: Ensemble model with highest accuracy and AUC.
  - **KNN**: Simple distance-based model, tested for comparison.
- Tuned hyperparameters with **Grid Search** for optimal performance.
- Evaluated models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 3. Feature Importance
- Service-related features such as **Online boarding**, **Inflight entertainment**, and **Seat comfort** emerged as the strongest predictors of passenger satisfaction.
- Operational features like delays had a weaker impact.
- Business travelers and passengers in Business Class reported higher satisfaction.

---

## Results

- **Random Forest** achieved the best performance:
  - Accuracy: 96%
  - Precision: 97%
  - Recall: 94%
  - F1-Score: 96%
  - AUC (ROC): 99%
- Decision Tree performed well with slightly lower metrics, offering better interpretability.
- Logistic Regression and KNN underperformed relative to ensemble methods.

---

## Applications

- Predicting passenger satisfaction for targeted customer service improvements.
- Identifying key features to enhance airline service quality.
- Supporting data-driven retention and loyalty programs.
- Informing operational and marketing strategies based on passenger behavior patterns.

---

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- jupyter

---

## Future Work

- Implement ensemble methods like Gradient Boosting or XGBoost for comparison.
- Explore deep learning models for more complex feature extraction.
- Incorporate external factors such as loyalty programs or macroeconomic trends.
- Develop an automated dashboard for real-time passenger satisfaction monitoring.

---

## Author

**Timothée Balland**  
UTC Compiègne/EDHEC Business School  
GitHub: [https://github.com/Timothee-Balland](https://github.com/Timothee-Balland)
