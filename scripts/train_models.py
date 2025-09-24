import numpy as np
import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


X_train = np.load('../data_processed/X_train.npy')
y_train = np.load('../data_processed/y_train.npy')


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Linear SVM': LinearSVC(random_state=42, max_iter=5000)
}


params = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5]
    },
    'Linear SVM': {
        'C': [0.01, 0.1, 1, 10, 100]
    }
}

best_models = {}


os.makedirs('../models', exist_ok=True)


for name in models:
    print(f"Tuning {name}...")
    clf = GridSearchCV(models[name], params[name], cv=3, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    best_models[name] = clf.best_estimator_
    print(f"Best params for {name}: {clf.best_params_}")


    model_path = f'../models/{name.lower().replace(" ", "_")}.joblib'
    joblib.dump(best_models[name], model_path)
    print(f"Saved {name} model to {model_path}")



