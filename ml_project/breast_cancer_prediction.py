# -*- coding: utf-8 -*-
"""BREAST CANCER PREDICTION.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12igRHdT2VSirIdHFmYji21bMhxTkATxP

Importing libraries
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('data.csv')
df

df.head()

df.isnull().sum()

df.dropna(axis=1,inplace=True)

df.isnull().sum()

df.drop(columns=['id'],inplace=True)

df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

x=df.drop(columns=['diagnosis'])
y=df['diagnosis']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_sc=scaler.fit_transform(x_train)
x_test_sc=scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Initialize the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB()
}
# Train and evaluate each classifier
for name, classifier in classifiers.items():
    classifier.fit(x_train_sc, y_train)
    y_pred = classifier.predict(x_test_sc)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy*100:.2f}%")

from sklearn.metrics import classification_report
for name, classifier in classifiers.items():
          print(f"\nClassification Report for {name}:\n")
          print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# Define parameter grids for each classifier
param_grids = {
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    "Support Vector Machine": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    "Random Forest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 5, 7]
    },
    "Naive Bayes": {} # No hyperparameters to tune for GaussianNB
}

# Perform hyperparameter tuning for each classifier
best_classifiers = {}
for name, classifier in classifiers.items():
    param_grid = param_grids.get(name, {})
    if param_grid:
        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(x_train_sc, y_train)
        best_classifiers[name] = grid_search.best_estimator_
        print(f"\nBest parameters for {name}: {grid_search.best_params_}")
        print(f"Best accuracy for {name}: {grid_search.best_score_:.2f}")
    else:
        best_classifiers[name] = classifier
        print(f"\nNo hyperparameters to tune for {name}")

for name, classifier in best_classifiers.items():
    classifier.fit(x_train_sc, y_train)
    y_pred = classifier.predict(x_test_sc)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy*100:.2f}%")


for name, classifier in best_classifiers.items():
    y_pred = classifier.predict(x_test_sc)
    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_test, y_pred))

import pickle
from sklearn.linear_model import LogisticRegression

# Example placeholder model training code:
model = LogisticRegression().fit(x_train, y_train)

# Save model
with open('logistic_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)