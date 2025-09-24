import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_test = np.load('../data_processed/X_test.npy')
y_test = np.load('../data_processed/y_test.npy')

models = {
    'Logistic Regression': joblib.load('../models/logistic_regression.joblib'),
    'Random Forest': joblib.load('../models/random_forest.joblib')
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'y_pred': y_pred
    })
    print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv('../output/performance_comparison.csv', index=False)
print('Evaluation complete. Results saved to output/performance_comparison.csv')



