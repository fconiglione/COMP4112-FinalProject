# COMP 4112 Introduction to Data Science
# Data Science Final Project
# Francesco Coniglione (st#1206780)

# Imports

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Reading in the dataset
file='american_bankruptcy.csv'
data = pd.read_csv(file)

# Converting the 'status_label' column to binary (alive = 1, failed = 0)
data["status_label"] = data["status_label"].apply(lambda x: 1 if x == "alive" else 0)

# Feature 1: Getting the ratio of current assets to total assets (known as liquidity)
def current_assets_ratio(row):
    return row['X1'] / row['X10'] if row['X10'] != 0 else 0

# Feature 2: Getting the ratio of total liabilities to profit (known as leverage)
def debt_to_equity_ratio(row):
    total_equity = row['X10'] - row['X17']
    return row['X17'] / total_equity if total_equity != 0 else 0

# Feature 3: Getting the return on assets (ROA)
def return_on_assets(row):
    return row['X6'] / row['X10'] if row['X10'] != 0 else 0

# Feature 4: Getting the ratio of net income to profit (known as profit-to-retained earnings)
def profit_margin(row):
    return row['X6'] / row['X16'] if row['X16'] != 0 else 0

# Feature 5: Finding the change in assets over time (known as asset growth)
def asset_growth(group):
    group = group.sort_values(by='year') # Ordering the entire company entries by year for consecutive growth calculation
    group['asset_growth'] = group['X10'].pct_change().fillna(0)
    return group

# Feature 6: Getting the debt to asset ratio
def debt_to_asset_ratio(row):
    return row['X17'] / row['X10'] if row['X10'] != 0 else 0

# Feature 7: Getting the current assets ratio (Current Assets / Current Liabilities)
def current_ratio(row):
    return row['X1'] / row['X14'] if row['X14'] != 0 else 0

# Feature 8: Getting what is known as the quick ratio (Current Assets - Inventory) / Current Liabilities
def quick_ratio(row):
    inventory = row['X5']
    current_assets = row['X1']
    return (current_assets - inventory) / row['X14'] if row['X14'] != 0 else 0

# Feature 9: Getting the asset turnover ratio (Net Sales / Total Assets)
def asset_turnover_ratio(row):
    return row['X9'] / row['X10'] if row['X10'] != 0 else 0

# Feature 10: Getting the Interest Coverage Ratio (Debt-Servicing Capacity)
def interest_coverage_ratio(row):
    return row['X12'] / row['X4'] if row['X4'] != 0 else 0

# Feature 11: Getting gross margin
def gross_margin_ratio(row):
    return row['X13'] / row['X9'] if row['X9'] != 0 else 0

# Applying the feature engineering functions
data['current_assets_ratio'] = data.apply(current_assets_ratio, axis=1)
data['debt_to_equity_ratio'] = data.apply(debt_to_equity_ratio, axis=1)
data['return_on_assets'] = data.apply(return_on_assets, axis=1)
data['profit_margin'] = data.apply(profit_margin, axis=1)
# Separateingeach company into a separate group and applying the asset growth function
data = data.groupby('company_name').apply(asset_growth)
data['debt_to_asset_ratio'] = data.apply(debt_to_asset_ratio, axis=1)
data['current_ratio'] = data.apply(current_ratio, axis=1)
data['quick_ratio'] = data.apply(quick_ratio, axis=1)
data['asset_turnover_ratio'] = data.apply(asset_turnover_ratio, axis=1)
data['interest_coverage_ratio'] = data.apply(interest_coverage_ratio, axis=1)
data['gross_margin_ratio'] = data.apply(gross_margin_ratio, axis=1)

# Dropping the 'company_name' column as they do nothing for the model
data.drop(columns=['company_name'], inplace=True)

# Defining features and the target variable
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18']] # Selecting the features
y = data['status_label'] # Target variable is obviously status_label

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Applying SMOTE technique for oversampling the minority class (the 'failed' companies) in the training set
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
classifierLR.fit(X_train_res, y_train_res)
pred = classifierLR.predict(X_test)
npYTest = np.array(y_test)
print("Logistic Regression test set score: {:.2f}".format(np.mean(pred == npYTest)))
confusion_matrix_lr = confusion_matrix(y_test, pred)
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix_lr)

# Visualization of the confusion matrices using heat maps

# def plot_confusion_matrix_with_labels(conf_matrix, title):
#     TN, FP, FN, TP = conf_matrix.ravel()

#     labels = np.array([[f'TN {TN}', f'FP {FP}'], [f'FN {FN}', f'TP {TP}']])

#     plt.figure(figsize=(6, 4))
#     sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues', cbar=False)
#     plt.title(title)
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.show()

# plot_confusion_matrix_with_labels(confusion_matrix_knn, "KNN Confusion Matrix")
# plot_confusion_matrix_with_labels(confusion_matrix_gnb, "Gaussian Naive Bayes Confusion Matrix")
# plot_confusion_matrix_with_labels(confusion_matrix_dtree, "Decision Tree Confusion Matrix")
# plot_confusion_matrix_with_labels(confusion_matrix_rndforest, "Random Forest Confusion Matrix")
# plot_confusion_matrix_with_labels(confusion_matrix_nn, "Neural Network Confusion Matrix")
# plot_confusion_matrix_with_labels(confusion_matrix_lr, "Logistic Regression Confusion Matrix")

# Accuracy comparisons using a bar chart

# accuracy_scores = [
#     np.mean(pred == y_test) for pred in [
#         classifierKNN.predict(X_test),
#         classifierGNB.predict(X_test),
#         classifierDTree.predict(X_test),
#         classifierRndForest.predict(X_test),
#         classifierNN.predict(X_test),
#         classifierLR.predict(X_test)
#     ]
# ]
# classifier_names = ['KNN', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Neural Network', 'Logistic Regression']

# plt.figure(figsize=(10, 5))
# sns.barplot(x=classifier_names, y=accuracy_scores)
# plt.xlabel('Classifiers')
# plt.ylabel('Accuracy Score')
# plt.title('Accuracy Comparison of Different Classifiers')
# plt.ylim(0, 1)
# plt.show()

# ROC curve

# from sklearn.metrics import roc_curve, auc

# def plot_roc_curve(classifier, X_test, y_test, title):
#     y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, "predict_proba") else classifier.decision_function(X_test)
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=f'{title} (AUC = {roc_auc:.2f})')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curves')
#     plt.legend(loc='lower right')

# plt.figure(figsize=(10, 7))
# plot_roc_curve(classifierKNN, X_test, y_test, "KNN")
# plot_roc_curve(classifierGNB, X_test, y_test, "Naive Bayes")
# plot_roc_curve(classifierDTree, X_test, y_test, "Decision Tree")
# plot_roc_curve(classifierRndForest, X_test, y_test, "Random Forest")
# plot_roc_curve(classifierNN, X_test, y_test, "Neural Network")
# plot_roc_curve(classifierLR, X_test, y_test, "Logistic Regression")
# plt.show()

"""
References

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://forecastegy.com/posts/feature-importance-in-random-forests/
https://scikit-learn.org/stable/modules/preprocessing.html
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

For SMOTE oversampling:
https://www.geeksforgeeks.org/smote-for-imbalanced-classification-with-python/

Code snippets were sourced from COMP 4112 lecture slides, assignment code, and the k-means-clustering-visual.py example file.
"""