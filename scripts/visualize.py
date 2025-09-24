import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib

df_results = pd.read_csv('../output/performance_comparison.csv')
y_test = np.load('../data_processed/y_test.npy')

models = {
    'Logistic Regression': '../models/logistic_regression.joblib',
    'Random Forest': '../models/random_forest.joblib'
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, model_file) in zip(axes, models.items()):
    model = joblib.load(model_file)
    X_test = np.load('../data_processed/X_test.npy')
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title(f'Confusion Matrix - {name}')
plt.tight_layout()
plt.savefig('../output/confusion_matrices.png')
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df_results['Model'], df_results['Accuracy'], color=['blue', 'green'])
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
plt.savefig('../output/accuracy_comparison.png')
plt.show()

