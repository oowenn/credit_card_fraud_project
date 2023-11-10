import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from joblib import dump
import numpy as np

# Load the processed dataset
df = pd.read_csv('data/processed_creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train a Random Forest to get feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, y_train_smote)

# Get feature importances and select the most important ones
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = X_train.columns[indices][:10]  # Selecting top 10 features
print(f"Selected features: {top_features}")

# Reduce training and test sets to top features
X_train_reduced = X_train_smote[top_features]
X_test_reduced = X_test[top_features]

# Define a smaller parameter grid because of the large dataset
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

# Randomized search with cross-validation
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42), 
    param_distributions=param_grid, 
    n_iter=10,  # Number of parameter settings sampled
    cv=3, 
    scoring='recall', 
    n_jobs=-1, 
    random_state=42
)
random_search.fit(X_train_reduced, y_train_smote)

# Retrieve the best estimator
best_rf_clf = random_search.best_estimator_

# Make predictions using the best Random Forest model
y_pred = best_rf_clf.predict(X_test_reduced)

# Print classification report and confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
model_filename = 'model/complex_selective_rfc.joblib'
dump(best_rf_clf, model_filename)

print(f"Model saved to {model_filename}")
 