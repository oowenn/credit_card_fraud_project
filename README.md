# Credit Card Fraud Detection Analysis

## Overview
This project involves the analysis and prediction of credit card fraud using a dataset from Kaggle. The objective is to gain exposure to the various components of data science. The project encompasses data cleaning, exploratory data analysis (EDA), and the development of a predictive model. If you want to explore this project on your own, you will need to download creditcard.csv from Kaggle [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/code) and put it in the `data` folder, since the file was too large to store on the repo.

## Data Cleaning
The data cleaning process involved handling missing values, normalizing the 'Amount' feature using `StandardScaler`, and removing any unnecessary or redundant features. Special care was taken to ensure that the data used for training the models did not leak information from the test set. You can find the relevant data cleaning figures in the `cleaning` folder.

- **Normalized 'Amount'**: The transaction 'Amount' feature was scaled to have a mean of zero and a standard deviation of one.
- **Handling Missing Data**: This dataset turned out to be pretty clean already, so I did not get the opportunity to handle missing data.
- **Feature Selection**: Irrelevant or less important features were dropped after careful analysis (i.e. 'Time').

## Exploratory Data Analysis (EDA)
The EDA involved examining the features to understand their distribution, detecting outliers, and identifying any correlations between features. You can find the relevant data cleaning figures in the `eda` folder.

- **Distribution Analysis**: Histograms and boxplots were used to visualize the distributions and identify any skewness in the features.
- **Correlation Matrix**: A heatmap was created to visualize the correlation between different features and the target variable. We were able to identify 3 features that seemed to have the strongest correlation to the target.
- **Outlier Detection**: The Interquartile Range (IQR) method was employed to detect and analyze outliers within the dataset.

## Model Development

The model development process was iterative and strategic, focusing on leveraging Random Forest Classifiers (RFC) due to their robustness and effectiveness in handling imbalanced datasets like ours. RFCs are an ensemble learning method that operate by constructing a multitude of decision trees during training time and outputting the class that is the mode of the classes (classification) of the individual trees. They are known for their high accuracy, ability to handle large datasets with higher dimensionality, and capability to model complex interactions between features.

### Model Variants Developed

Four variants of RFC models were developed to identify the most effective approach:

### Base RFC Model
The `base_rfc` model served as a benchmark, using all available features. It set a high accuracy baseline, which was anticipated due to the imbalanced nature of the dataset, yet highlighted the need for more nuanced performance metrics.

### Complex RFC Model
The `complex_rfc` model introduced hyperparameter tuning with a comprehensive grid search approach. This method aimed to systematically work through multiple combinations of parameter tunes, cross-validating as it went to determine which tune gave the best performance. Unfortunately, the model did not perform as expected, likely due to the large search space and the extensive computational demand which might have led to overfitting.

### Selective RFC Model
The `selective_rfc` model was a pivot towards a feature-focused approach, where the model was trained using only features deemed most predictive from the EDA. This decision was based on the premise that a model with fewer, more relevant features can often outperform a model with more features that may include noise.

### Complex Selective RFC Model
The `complex_selective_rfc` combined the feature-focused approach with hyperparameter tuning but utilized a more conservative parameter grid, balancing the benefits of optimization against the constraints of computational efficiency. This model demonstrated an improved recall, underscoring its potential utility in scenarios where the cost of false negatives is high.

### Decision Making and Tuning Process

The decision to use RFCs was driven by their inherent ability to perform feature selection on the fly and provide a measure of feature importance as part of the model output. This feature was particularly useful in both reducing dimensionality and honing in on the most significant predictors of fraud.

The tuning process involved narrowing down the hyperparameters that significantly impact the model's ability to generalize. Parameters like the number of trees (`n_estimators`), the depth of the trees (`max_depth`), and the minimum number of samples required to split a node (`min_samples_split`) were adjusted to find the optimal balance between bias and variance.

The performance of each model was meticulously recorded, compared, and contrasted based on several metrics, including precision, recall, and the F1-score, to ensure a holistic view of model efficacy beyond mere accuracy.

### Insights

The iterative modeling process elucidated key insights into the delicate balance between model complexity, feature selection, and evaluation metrics. It became evident that in fraud detection, where the cost of false negatives can be substantial, models must be tuned with a focus on recall without overly compromising on precision.

### Running the Models

Each model is self-contained within a script corresponding to its name:

- `base_rfc.py`
- `complex_rfc.py`
- `selective_rfc.py`
- `complex_selective_rfc.py`

To run any model, ensure the dataset is placed in the root directory and execute the script. Each model will save its trained version for future use.



## Results
The model demonstrated excellent performance on the testing set, with a high recall rate indicating a strong ability to identify fraudulent transactions. Precision, accuracy, and F1-score were also considered to ensure a balanced model.

- **Confusion Matrix**: Used to evaluate the number of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provided a detailed report including precision, recall, and F1-score for each class.

## Conclusion
The predictive model showed promising results in detecting fraudulent transactions. Future work may include testing alternative models, feature engineering, and deploying the model as part of a real-time fraud detection system.

## How to Run the Project
1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the data cleaning script: `python data_cleaning.py`.
4. Execute the EDA notebook: `jupyter notebook eda.ipynb`.
5. Train the model: `python model_training.py`.

Ensure that the dataset files are located in the correct directory as expected by the scripts.

---

For any additional information or inquiries, please contact anchengowenshi@gmail.com.

