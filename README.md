# Airlines Customer Satisfaction Analysis

## Overview
This project explores customer satisfaction in the airline industry using the **Airline Passenger Satisfaction** dataset. The study combines descriptive analysis and machine learning techniques to identify key drivers of satisfaction and predict whether a passenger is satisfied or neutral/dissatisfied.

The dataset contains **103,904 instances** and **25 features**, including demographic information, service ratings, and operational metrics. The goal of this project is twofold:

1. **Descriptive Analysis**: Identify patterns, relationships, and key factors influencing passenger satisfaction.
2. **Machine Learning**: Build predictive models to classify passengers into satisfied or neutral/dissatisfied categories and evaluate model performance.

---

## Project Structure

Final_Project/
│
├── Airline_Satisfaction-Analysis.ipynb # Main Jupyter Notebook with full analysis
├── data/ # Folder containing datasets
│ ├── train.csv
│ └── test.csv
├── README.md # Project overview and instructions
└── figures/ # Optional folder for saved figures

---

## Table of Contents of the Analysis

### 1. Introduction
- Overview of dataset and objectives
- Importance of customer satisfaction in the airline industry

### 2. Part I: Descriptive Analysis
- **Dataset Overview**: 25 features including numerical and categorical variables
- **Univariate Analysis**: Summary statistics, histograms, and count plots
- **Bivariate Analysis**: Correlation matrix, scatterplots, boxplots, t-tests, and chi-square tests
- **Key Takeaways**:
  - Service quality features (Online boarding, Inflight entertainment, Seat comfort) strongly impact satisfaction.
  - Operational variables (delays) have a weaker effect.
  - Business travelers and passengers in Business Class report higher satisfaction.

### 3. Part II: Machine Learning Techniques
- **Models Applied**:
  - Decision Tree
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
- **Model Evaluations**:
  - Decision Tree: Accuracy 95%, F1 94%, AUC 94%
  - Logistic Regression: Accuracy 88%, F1 86%, AUC 93%
  - Random Forest: Accuracy 96%, F1 96%, AUC 99%
  - KNN: Accuracy 60%, F1 50%, AUC 61%
- **Hyperparameter Optimization**:
  - Grid Search applied to Decision Tree and Random Forest
- **Key Insights**:
  - Random Forest achieved the best overall performance.
  - Service-related features were the most important predictors.
  - KNN underperformed due to sensitivity to feature scaling and data imbalance.

### 4. Conclusion
- Service-related features are the strongest drivers of passenger satisfaction.
- Random Forest is the most effective model for prediction.
- Future work could include ensemble methods like Gradient Boosting or deep learning approaches.

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Timothee-Balland/Airlines-Customer-Satisfaction-Report.git
cd Final_Project
Install required packages:
pip install -r requirements.txt
Open the Jupyter Notebook:
jupyter notebook Airline_Satisfaction-Analysis.ipynb
Follow the notebook to explore the analysis, visualizations, and machine learning models.
Requirements
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
License
This project is for educational purposes. You may use it for learning or research, but redistribution or commercial use is not allowed without permission.
Author
Timothée Balland
BBA3 Student, EDHEC Business School
GitHub: https://github.com/Timothee-Balland
