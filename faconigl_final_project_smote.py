# COMP 4112 Introduction to Data Science
# Data Science Final Project
# Francesco Coniglione (st#1206780)

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Reading the dataset
data = pd.read_csv('american_bankruptcy.csv')

# Convert the status_label column to binary (alive = 1, failed = 0)
data["status_label"] = data["status_label"].apply(lambda x: 1 if x == "alive" else 0)

# Feature 1: Getting the ratio of current assets to total assets (liquidity)
def current_assets_ratio(row):
    return row['X1'] / row['X10'] if row['X10'] != 0 else 0

# Feature 2: Getting the ratio of total liabilities to profit (leverage)
def debt_to_equity_ratio(row):
    return row['X17'] / row['X15'] if row['X15'] != 0 else 0

# Feature 3: Getting the return on assets (ROA)
def return_on_assets(row):
    return row['X6'] / row['X10'] if row['X10'] != 0 else 0

# Feature 4: Getting the ratio of net income to profit (profit-to-retained earnings)
def net_income_to_profit(row):
    return row['X6'] / row['X15'] if row['X15'] != 0 else 0

# Feature 5: Finding the change in assets over time (asset growth)
def asset_growth(group):
    group = group.sort_values(by='year') # Order the entire company entries by year for consecutive growth calculation
    group['asset_growth'] = group['X10'].pct_change().fillna(0)
    return group

# Feature 6: Getting the debt to asset ratio
def debt_to_asset_ratio(row):
    return row['X17'] / row['X10'] if row['X10'] != 0 else 0

# Feature 7: Get net income (X6)
def net_income(row):
    return row['X6']

# Feature 8: Get market value (X8)
def market_value(row):
    return row['X8']

# Apply the feature engineering functions
data['current_assets_ratio'] = data.apply(current_assets_ratio, axis=1)
data['debt_to_equity_ratio'] = data.apply(debt_to_equity_ratio, axis=1)
data['return_on_assets'] = data.apply(return_on_assets, axis=1)
data['net_income_to_profit'] = data.apply(net_income_to_profit, axis=1)
# Separate each company into a separate group and apply the asset growth function
data = data.groupby('company_name').apply(asset_growth)
data['debt_to_asset_ratio'] = data.apply(debt_to_asset_ratio, axis=1)
data['net_income'] = data.apply(net_income, axis=1)
data['market_value'] = data.apply(market_value, axis=1)

# Drop the 'company_name' and 'Company Name' columns as they are not useful for modeling
data.drop(columns=['company_name'], inplace=True)

# Define features and target variable
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18']] # Selecting the features
y = data['status_label'] # Target variable is obviously status_label

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE for oversampling the minority class in the training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# KNN Classifier
classifierKNN = KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train_res, y_train_res)
pred = classifierKNN.predict(X_test)
npYTest = np.array(y_test)
print("K-Nearest Neighbor test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_knn = confusion_matrix(y_test, pred)
print("KNN Confusion Matrix:")
print(confusion_matrix_knn)

# Decision Tree Classifier
classifierDTree = DecisionTreeClassifier(random_state=42)
classifierDTree.fit(X_train_res, y_train_res)
pred = classifierDTree.predict(X_test)
npYTest = np.array(y_test)
print("Decision Tree test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_dtree = confusion_matrix(y_test, pred)
print("Decision Tree Confusion Matrix:")
print(confusion_matrix_dtree)

# Random Forest Classifier
classifierRndForest = RandomForestClassifier(random_state=42)
classifierRndForest.fit(X_train_res, y_train_res)
pred = classifierRndForest.predict(X_test)
npYTest = np.array(y_test)
print("Random Forest test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_rndforest = confusion_matrix(y_test, pred)
print("Random Forest Confusion Matrix:")
print(confusion_matrix_rndforest)

# Logistic Regression
classifierLR = LogisticRegression(random_state=42)
classifierLR.fit(X_train_res, y_train_res)
pred = classifierLR.predict(X_test)
npYTest = np.array(y_test)
print("Logistic Regression test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_lr = confusion_matrix(y_test, pred)
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix_lr)