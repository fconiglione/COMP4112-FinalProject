# COMP 4112 Introduction to Data Science
# Data Science Final Project
# Francesco Coniglione (st#1206780)

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the dataset
file = 'american_bankruptcy.csv'
data = pd.read_csv(file)

# Drop the 'Company' column as it does nothing for the model
data = data.drop(columns=['company_name'])

# Convert the status_label column to binary (alive = 1, failed = 0)
data["status_label"] = data["status_label"].apply(lambda x: 1 if x == "alive" else 0)

# Split the dataset into features and target
X = data.drop(columns=["status_label"])
y = data["status_label"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Logistic Regression Classifier with class weighting to handle imbalanced data
regr = LogisticRegression(class_weight='balanced', max_iter=1000)
regr.fit(X_train, y_train)

# Train a Random Forest Classifier for comparison
rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_clf.fit(X_train, y_train)

# Predict the target values using both models
y_pred_regr = regr.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)

# Evaluate Logistic Regression Model
print("Logistic Regression Results:")
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred_regr))
print("Precision: %.2f" % precision_score(y_test, y_pred_regr))
print("Recall: %.2f" % recall_score(y_test, y_pred_regr))
print("F1 Score: %.2f" % f1_score(y_test, y_pred_regr))
print("Classification Report:")
print(classification_report(y_test, y_pred_regr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_regr))

# Evaluate Random Forest Model
print("\nRandom Forest Results:")
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred_rf))
print("Precision: %.2f" % precision_score(y_test, y_pred_rf))
print("Recall: %.2f" % recall_score(y_test, y_pred_rf))
print("F1 Score: %.2f" % f1_score(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))