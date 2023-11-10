import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

### load dataset
df = pd.read_csv('data/creditcard.csv')

### Understanding/Summarizing the Data
# Print first few rows of the dataframe
print("----- First Few Rows -----")
print(df.head())
print("----- End of Section -----\n")

# Summary statistics for numerical features
print("----- Summary Statistics -----")
print(df.describe())
print("----- End of Section -----\n")

# Check for missing values
print("----- Missing Values -----")
print(df.isnull().sum())
print("----- End of Section -----\n")

# Check the class distribution (fraud vs. non-fraud)
print("----- Checking Class Distribution -----")
print(df['Class'].value_counts())
print("----- End of Section -----\n")

# Visualize the class imbalance
print("----- Visualizing Class Distribution -----")
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
print('Plotting Class Distribution')
plt.show()
plt.clf()
print("----- End of Section -----\n")

### Scaling Values
# Normalize 'Amount'
print("----- Normalizing 'Amount' -----")
scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
print("Added 'NormalizedAmount' column to data")

# Depecting the difference between 'Amount' and 'NormalizedAmount'
# Plotting the original 'Amount' feature
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
sns.histplot(df['Amount'], bins=100, ax=ax[0], kde=True)
ax[0].set_title('Distribution of Original Amount')
ax[0].set_xlim([0, 2500])  # Limit x-axis for better visibility if necessary
ax[0].set_xlabel('Amount')

# Plotting the 'NormalizedAmount' after scaling
sns.histplot(df['NormalizedAmount'], bins=100, ax=ax[1], kde=True)
ax[1].set_title('Distribution of Scaled Amount')
ax[1].set_xlabel('Normalized Amount')
filename = 'cleaning/amount_vs_normalizedamount.png'
plt.savefig(filename)
plt.clf()
print('Saved', filename)
print("----- End of Section -----\n")

### Feature Selection
# Drop the original 'Time' and 'Amount' columns
print("----- Selecting Relevant Features -----")
df = df.drop(['Time', 'Amount'], axis=1)
print("Dropped 'Time' and 'Amount' columns from data")
print("----- End of Section -----\n")

### Preparing for Modeling
print("----- Selecting Relevant Features -----")
# Features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# During training, we will use SMOTE to fix the imbalance of the Class distribution in this dataset
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print("Using SMOTE to fix the imbalance of the 'Class' distribution")

# Check the class distribution after SMOTE
# print(pd.Series(y_res).value_counts())

# Depicting the effects of SMOTE on 'Class' Distribution
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Class distribution before SMOTE
sns.countplot(x='Class', data=df, ax=ax[0])
ax[0].set_title('Class Distribution Before SMOTE')

# Class distribution after SMOTE
sns.countplot(x=y_res, ax=ax[1])
ax[1].set_title('Class Distribution After SMOTE')
filename = 'cleaning/class_distribution_before_vs_after.png'
plt.savefig(filename)
plt.clf()
print('Saved', filename)
print("----- End of Section -----\n")

### Saving Processed Data
print("----- Saving Processed Data -----")
# Saving the processed data to a new CSV file
filename = 'data/processed_creditcard.csv'
df.to_csv(filename, index=False)
print('Saved', filename)
print("----- End of Section -----\n")
