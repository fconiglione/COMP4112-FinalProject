# COMP 4112 Introduction to Data Science
# Data Science Final Project
# Francesco Coniglione (st#1206780)

import os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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