Machine Learning Driven Structure–Property Correlation for Alloy Phase Prediction
Overview

This project applies Machine Learning techniques to predict the phase structure of High Entropy Alloys (HEAs) based on compositional and thermodynamic features. HEAs are advanced metallic materials whose mechanical properties depend heavily on their phase structure such as FCC, BCC, FCC+BCC, and Intermetallic phases.

Traditional experimental methods for determining alloy phases are slow, expensive, and resource-intensive. This project demonstrates how Machine Learning models can accurately predict alloy phases quickly and efficiently, enabling faster materials design and optimization.

Objectives

Predict phase structure of High Entropy Alloys using Machine Learning

Compare performance of multiple ML algorithms

Identify most important features influencing alloy phase prediction

Evaluate model performance using standard classification metrics

Determine the best model for accurate and reliable prediction

Dataset Description
Features Used

The dataset includes compositional, thermodynamic, and synthesis features such as:

Number of Elements in Alloy

Valence Electron Concentration (VEC)

Atomic Size Difference

Electronegativity Difference

Enthalpy of Mixing (ΔHmix)

Entropy of Mixing (ΔSmix)

Synthesis Route

Phase Structure (Target Variable)

Data Preprocessing

Removed irrelevant columns

Handled missing values

Encoded categorical variables

Standardized numerical features

Balanced dataset using SMOTE (for SVM)

Machine Learning Models Used

Four Machine Learning models were implemented and compared:

1. Support Vector Machine (SVM)

Kernel: RBF

Training Accuracy: ~75%

Test Accuracy: ~78%

Used StandardScaler and SMOTE

Feature importance analyzed using permutation importance

2. XGBoost Classifier

Training Accuracy: High

Test Accuracy: ~85%

Strong classification performance

Efficient and scalable algorithm

3. Random Forest Classifier

Number of trees: 100

Training Accuracy: ~92%

Test Accuracy: ~85%

Provided strong feature importance insights

Robust and reliable performance

4. Neural Network (MLP Classifier)

Architecture: 2 hidden layers (100, 50 neurons)

Activation: ReLU

Optimizer: Adam

Training Accuracy: ~88%

Test Accuracy: ~89%

Best performing model overall

Methodology
Step 1: Data Preparation

Load dataset

Clean and preprocess data

Encode categorical variables

Handle missing values

Step 2: Feature Engineering

Select relevant features

Standardize numerical features

Split dataset into training and testing sets

Step 3: Model Training

Train SVM, XGBoost, Random Forest, and Neural Network models

Apply appropriate preprocessing techniques

Step 4: Model Evaluation

Models evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Feature Importance

Results Summary
Model	Test Accuracy	Performance
Neural Network	89%	Best
XGBoost	85%	Excellent
Random Forest	85%	Excellent
SVM	78%	Moderate
Conclusion

The Neural Network (MLP Classifier) achieved the best overall performance and demonstrated the highest accuracy and reliability for predicting alloy phases.

XGBoost and Random Forest also performed very well and are strong alternatives with lower computational complexity.

Support Vector Machine performed comparatively lower and may not be ideal for this dataset.

This project demonstrates the effectiveness of Machine Learning in accelerating material discovery and reducing dependence on expensive experimental methods.

Technologies Used

Python

Scikit-learn

XGBoost

NumPy

Pandas

Matplotlib

Seaborn

Project Structure
project-folder/
│
├── data/
│   └── alloy_dataset.csv
│
├── models/
│   ├── svm_model.py
│   ├── xgboost_model.py
│   ├── random_forest_model.py
│   └── neural_network_model.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
└── README.md

Applications

Materials Informatics

Alloy Design Optimization

Predictive Materials Science

Accelerated Materials Discovery

Team Members

Rishitha C – Support Vector Machine

Divya Rithanya S – Random Forest

Karthikeya Y – XGBoost

Kaushal S – Neural Networks
