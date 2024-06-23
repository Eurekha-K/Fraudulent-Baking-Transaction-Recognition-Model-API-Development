# Fraudulent-Baking-Transaction-Recognition-
Classifying the Fradulent and Non-Fradulent transactions using  Classification ALgorithms


Fraudulent Banking Operations Recognition - Machine Learning Model
This project investigates fraudulent banking operations using machine learning algorithms. It references the published journal "Developing AI-based Fraud Detection Systems for Banking and Finance" ([reference link]).

# Dataset:

Credit Card transactions dataset (details anonymized due to security concerns)
Data is already standardized.
Methodology:

# Feature Engineering:

## Data Cleaning: No missing values present due to pre-processing.
Feature Selection: All features except "time" are included.
Dimensionality Reduction: PCA (Principal Component Analysis) is not applied as sufficient features remain after initial processing.
Feature Scaling: StandardScaler is applied to the "Amount" column due to its unit (dollars).
Imbalanced Data Handling:

The dataset is imbalanced (unequal distribution of fraudulent and legitimate transactions).
Undersampling is used to balance the data (reducing the majority class).
# Data Splitting:
Independent features (predictors) and dependent feature (target variable) are separated.
# Training and Testing Split:
Data is split into training (80%) and testing (20%) sets.
Machine Learning Model Selection:
Logistic Regression

Decision Tree Classifier

Random Forest

Support Vector Machine
# Choosing the Best Model
Random Forest achieved the best performance among the evaluated algorithms with accuracy of 96 % .

