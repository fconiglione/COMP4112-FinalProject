# COMP 4112 Introduction to Data Science
# Data Science Final Project
# Francesco Coniglione (st#1206780)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Reading the dataset
data = pd.read_csv('american_bankruptcy.csv')

# Feature 1: Ratio of Current Assets to Total Assets (Liquidity)
def current_assets_ratio(row):
    return row['X1'] / row['X10'] if row['X10'] != 0 else 0

# Feature 2: Debt-to-Equity ratio (financial leverage)
def debt_to_equity_ratio(row):
    return row['X17'] / row['X15'] if row['X15'] != 0 else 0

# Feature 3: Return on Assets (ROA)
def roa(row):
    return row['X6'] / row['X10'] if row['X10'] != 0 else 0

# Feature 4: Return on Equity (ROE)
def roe(row):
    return row['X6'] / row['X15'] if row['X15'] != 0 else 0

# Feature 5: Growth of Total Assets (e.g., change in total assets from one year to the next)
def asset_growth(row, previous_row):
    return (row['X10'] - previous_row['X10']) / previous_row['X10'] if previous_row['X10'] != 0 else 0

# Feature 7: Interaction between Total Assets and Total Debt (leverage indicator)
def asset_debt_interaction(row):
    return row['X10'] * row['X11']

# Apply the feature engineering functions
data['current_assets_ratio'] = data.apply(current_assets_ratio, axis=1)
data['debt_to_equity'] = data.apply(debt_to_equity_ratio, axis=1)
data['roa'] = data.apply(roa, axis=1)
data['roe'] = data.apply(roe, axis=1)

# Asset growth: Applying it based on previous row, assuming rows are ordered by year
data['asset_growth'] = data.apply(lambda row: asset_growth(row, data.iloc[data.index.get_loc(row.name)-1] if row.name > 0 else row), axis=1)

data['asset_debt_interaction'] = data.apply(asset_debt_interaction, axis=1)

# Drop the 'company_name' and 'Company Name' columns as they are not useful for modeling
data.drop(columns=['company_name'], inplace=True)

# Define features and target variable
X = data.drop(columns=["status_label", "year"])  # Features (excluding target and year)
y = data["status_label"]  # Target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# KNN Classifier
classifierKNN = KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train, y_train)
pred = classifierKNN.predict(X_test)
print("K-Nearest Neighbor Accuracy: {:.2f}".format(np.mean(pred == y_test)))

# Decision Tree Classifier
classifierDTree = DecisionTreeClassifier(random_state=42)
classifierDTree.fit(X_train, y_train)
pred = classifierDTree.predict(X_test)
print("Decision Tree Accuracy: {:.2f}".format(np.mean(pred == y_test)))

# Random Forest Classifier
classifierRndForest = RandomForestClassifier(random_state=42)
classifierRndForest.fit(X_train, y_train)
pred = classifierRndForest.predict(X_test)
print("Random Forest Accuracy: {:.2f}".format(np.mean(pred == y_test)))

# Naive Bayes Classifier
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)
pred = classifierNB.predict(X_test)
print("Naive Bayes Accuracy: {:.2f}".format(np.mean(pred == y_test)))

# Logistic Regression
classifierLR = LogisticRegression(random_state=42)
classifierLR.fit(X_train, y_train)
pred = classifierLR.predict(X_test)
print("Logistic Regression Accuracy: {:.2f}".format(np.mean(pred == y_test)))