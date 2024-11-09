# COMP 4112 Introduction to Data Science
# Data Science Final Project
# Francesco Coniglione (st#1206780)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Reading the dataset
data = pd.read_csv('american_bankruptcy.csv')

# Convert the status_label column to binary (alive = 1, failed = 0)
data["status_label"] = data["status_label"].apply(lambda x: 1 if x == "alive" else 0)

# Feature 1: Getting the ratio of current assets to total assets (liquidity)
def current_assets_ratio(row):
    return row['X1'] / row['X10'] if row['X10'] != 0 else 0

# Feature 2: Getting the ratio of total liabilities to profit (leverage)
def debt_to_equity_ratio(row):
    total_equity = row['X10'] - row['X17']
    return row['X17'] / total_equity if total_equity != 0 else 0

# Feature 3: Getting the return on assets (ROA)
def return_on_assets(row):
    return row['X6'] / row['X10'] if row['X10'] != 0 else 0

# Feature 4: Getting the ratio of net income to profit (profit-to-retained earnings)
def profit_margin(row):
    return row['X6'] / row['X16'] if row['X16'] != 0 else 0

# Feature 5: Finding the change in assets over time (asset growth)
def asset_growth(group):
    group = group.sort_values(by='year') # Order the entire company entries by year for consecutive growth calculation
    group['asset_growth'] = group['X10'].pct_change().fillna(0)
    return group

# Feature 6: Getting the debt to asset ratio
def debt_to_asset_ratio(row):
    return row['X17'] / row['X10'] if row['X10'] != 0 else 0

# Feature 7: Current assets ratio (Current Assets / Current Liabilities)
def current_ratio(row):
    return row['X1'] / row['X14'] if row['X14'] != 0 else 0

# Feature 8: Quick ratio (Current Assets - Inventory) / Current Liabilities
def quick_ratio(row):
    inventory = row['X5']
    current_assets = row['X1']
    return (current_assets - inventory) / row['X14'] if row['X14'] != 0 else 0

# Feature 9: Asset turnover ratio (Net Sales / Total Assets)
def asset_turnover_ratio(row):
    return row['X9'] / row['X10'] if row['X10'] != 0 else 0

# Feature 10: Interest Coverage Ratio (Debt-Servicing Capacity)
def interest_coverage_ratio(row):
    return row['X12'] / row['X4'] if row['X4'] != 0 else 0

# Feature 11: Gross margin
def gross_margin_ratio(row):
    return row['X13'] / row['X9'] if row['X9'] != 0 else 0

# Apply the feature engineering functions
data['current_assets_ratio'] = data.apply(current_assets_ratio, axis=1)
data['debt_to_equity_ratio'] = data.apply(debt_to_equity_ratio, axis=1)
data['return_on_assets'] = data.apply(return_on_assets, axis=1)
data['profit_margin'] = data.apply(profit_margin, axis=1)
# Separate each company into a separate group and apply the asset growth function
data = data.groupby('company_name').apply(asset_growth)
data['debt_to_asset_ratio'] = data.apply(debt_to_asset_ratio, axis=1)
data['current_ratio'] = data.apply(current_ratio, axis=1)
data['quick_ratio'] = data.apply(quick_ratio, axis=1)
data['asset_turnover_ratio'] = data.apply(asset_turnover_ratio, axis=1)
data['interest_coverage_ratio'] = data.apply(interest_coverage_ratio, axis=1)
data['gross_margin_ratio'] = data.apply(gross_margin_ratio, axis=1)

# Drop the 'company_name' and 'Company Name' columns as they are not useful for modeling
data.drop(columns=['company_name'], inplace=True)

# Define features and target variable
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18']] # Selecting the features
y = data['status_label'] # Target variable is obviously status_label

# Scaling the data due to the following error:
"""
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# KNN Classifier
classifierKNN = KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train, y_train)
pred = classifierKNN.predict(X_test)
npYTest = np.array(y_test)
print("K-Nearest Neighbor test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_knn = confusion_matrix(y_test, pred)
print("KNN Confusion Matrix:")
print(confusion_matrix_knn)

# Gaussian Naive Bayes Classifier
classifierGNB = GaussianNB()
classifierGNB.fit(X_train, y_train)
pred = classifierGNB.predict(X_test)
print("Gaussian Naive Bayes test set score: {:.2f}".format(np.mean(pred == y_test)))
confusion_matrix_gnb = confusion_matrix(y_test, pred)
print("Gaussian Naive Bayes Confusion Matrix:")
print(confusion_matrix_gnb)

# Decision Tree Classifier
classifierDTree = DecisionTreeClassifier(random_state=42)
classifierDTree.fit(X_train, y_train)
pred = classifierDTree.predict(X_test)
npYTest = np.array(y_test)
print("Decision tree test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_dtree = confusion_matrix(y_test, pred)
print("Decision Tree Confusion Matrix:")
print(confusion_matrix_dtree)

# Random Forest Classifier
classifierRndForest = RandomForestClassifier(random_state=42)
classifierRndForest.fit(X_train, y_train)
pred = classifierRndForest.predict(X_test)
npYTest = np.array(y_test)
print("Random forest test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_rndforest = confusion_matrix(y_test, pred)
print("Random Forest Confusion Matrix:")
print(confusion_matrix_rndforest)

"""
# Calculating feature importance from https://forecastegy.com/posts/feature-importance-in-random-forests/
importances = classifierRndForest.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)
"""

# Neural Network Classifier
classifierNN = MLPClassifier(random_state=42)
classifierNN.fit(X_train, y_train)
pred = classifierNN.predict(X_test)
npYTest = np.array(y_test)
print("Neural Network test set score: {:.2f}".format(np.mean(pred == npYTest))
)
confusion_matrix_nn = confusion_matrix(y_test, pred)
print("Neural Network Confusion Matrix:")
print(confusion_matrix_nn)

# Logistic Regression
classifierLR = LogisticRegression(random_state=42)
classifierLR.fit(X_train, y_train)
pred = classifierLR.predict(X_test)
npYTest = np.array(y_test)
print("Logistic Regression test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_lr = confusion_matrix(y_test, pred)
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix_lr)

"""
# Checking the correlation of the features with the target variable
correlation_matrix = data.corr()
print(correlation_matrix['status_label'].sort_values(ascending=False))
"""