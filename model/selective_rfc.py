import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load the processed dataset
df = pd.read_csv('data/processed_creditcard.csv')

### Selecting features with good correlations
correlation_matrix = df.corr()

# Get the absolute correlation values of each feature with 'Class'
correlations_with_target = correlation_matrix['Class'].abs().sort_values(ascending=False)

# Select a threshold for selecting features
threshold = 0.1

# Get the list of features that meet the threshold criteria
selected_features = correlations_with_target[correlations_with_target > threshold].index.tolist()

print(f"Selected features: {selected_features}")

df_selected = df[selected_features]

# Separate features and target
X = df_selected.drop('Class', axis=1)
y = df_selected['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize the Random Forest Classifier with default parameters
rf_clf = RandomForestClassifier(random_state=42)

# Train the model
rf_clf.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = rf_clf.predict(X_test)

# Print classification report and confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
model_filename = 'model/selective_rfc.pkl'
dump(rf_clf, model_filename)

print(f"Model saved to {model_filename}")
