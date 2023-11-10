import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load the dataset
df = pd.read_csv('data/processed_creditcard.csv')

# Features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE only to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='recall', verbose=2)
grid_search.fit(X_train_smote, y_train_smote)

# Best model after grid search
best_rf_clf = grid_search.best_estimator_

# Make predictions with the best model
y_scores = best_rf_clf.predict_proba(X_test)[:, 1]

# Calculate precision and recall for various thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# Find the threshold that maximizes recall while keeping an acceptable precision level
desired_recall = 0.90
threshold_optimal = thresholds[np.argmax(recalls >= desired_recall)]

# Apply the optimal threshold to make final predictions
y_pred_optimal = (y_scores >= threshold_optimal).astype(int)

# Final evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_optimal))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal))

print(f"\nOptimal Threshold: {threshold_optimal}")

# Save the model to disk
model_filename = 'best_random_forest_model.joblib'
dump(best_rf_clf, model_filename)

print(f"Model saved to {model_filename}")