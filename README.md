# AutoAlgoRecommender

AutoAlgoRecommender is a Python framework that automates the selection of the best machine learning algorithm for your dataset. It preprocesses data, evaluates multiple models, and recommends the top-performing algorithm based on accuracy (for classification) or mean squared error (for regression).

## Features
- Detects whether the task is **classification** or **regression**.
- Handles **data preprocessing** (scaling, encoding, and splitting).
- Evaluates models like **Logistic Regression**, **Decision Trees**, **Random Forests**, **SVM**, and **k-NN**.
- Recommends the top-performing model based on **accuracy** or **MSE**.

## Installation

### Clone the Repository

git clone https://github.com/vikash029/AutoAlgoRecommender.git


cd AutoAlgoRecommender

pip install -r requirements.txt


import pandas as pd
from automl_selector import AutoAlgoSelector

# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Get the recommended algorithm
best_model_name, best_model = AutoAlgoSelector.recommend_model(data, target_column="target")
print(f"Best model: {best_model_name}")


Supported Algorithms
Classification: Logistic Regression, Decision Trees, Random Forests, SVM, k-NN
Regression: Linear Regression, Decision Trees, Random Forest Regressor, SVR, k-NN Regressor
Contributing
Fork the repository, create a new branch for your feature/fix, make changes, run tests, and submit a pull request.


Issues & Future Improvements
Support for advanced algorithms (e.g., XGBoost, LightGBM, CatBoost).
Hyperparameter tuning and model optimization.
Thank you for using AutoAlgoRecommender! 
