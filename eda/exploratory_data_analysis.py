import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### Load the dataset preprocessed from cleaning.py
df = pd.read_csv('data/processed_creditcard.csv')

### Univariate Analysis
# Histogram for all features
print("----- Univariate Analysis -----")
df.hist(bins=50, figsize=(20,15))
filename = 'eda/features_hist.png'
plt.savefig(filename)
print('Saved', filename)
plt.clf()
# # Count plots for categorical features, but there doesn't seem to be any
# for column in df.select_dtypes(include=['O']).columns:
#     sns.countplot(y=column, data=df)
#     plt.show()
print("----- End of Section -----\n")

### Bivariate Analysis
# Boxplots for features by class
print("----- Bivariate Analysis -----")
for col in df.drop('Class', axis=1).columns:
    sns.boxplot(x='Class', y=col, data=df)
    plt.title(f'Class vs {col}')
    filename = 'eda/' + col + '_vs_Class.png'
    plt.savefig(filename)
    print('Saved', filename)
    plt.clf()
# # Cross-tabulation for categorical features vs target, but there doesn't seem to be any
# for column in df.select_dtypes(include=['O']).columns:
#     print(pd.crosstab(df[column], df['Class']))
print("----- End of Section -----\n")

### Correlation Analysis
# Correlation matrix
print("----- Correlation Analysis -----")
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
filename = 'eda/correlation_matrix.png'
plt.savefig(filename)
print('Saved', filename)
plt.clf()
print("----- End of Section -----\n")

### Multivariate Analysis
# Pairplot for a subset of features
# We select V12, V14, and V17 based on the correlation analysis 
print("----- Multivariate Analysis -----")
print('Creating pairplot based on the 3 features that we saw the most correlation with Class. This may take a while...')
sns.pairplot(df, vars=['V12', 'V14', 'V17', 'Class'], hue='Class')
filename = 'eda/v12_v14_v17_class_pairplot.png'
plt.savefig(filename)
print('Saved', filename)
plt.clf()
print("----- End of Section -----\n")

### Outlier Detection
# Detecting outliers using IQR
print("----- Outlier Percentages -----")
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
# Calculate the outlier mask (True for each outlier)
outliers_mask = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

# Count the number of outliers for each feature and convert it to percentage
outliers_percentage = (outliers_mask.sum() / len(df)) * 100
print(outliers_percentage)
print("----- End of Section -----\n")