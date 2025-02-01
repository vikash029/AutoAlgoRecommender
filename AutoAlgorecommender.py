import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def recommend_model(data, target_column):
    # Identify the problem type: classification or regression
    if data[target_column].dtype in ['int64', 'float64']:
        problem_type = 'regression'
    else:
        problem_type = 'classification'

    # Preprocess the data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Handle categorical variables
    if problem_type == 'classification':
        y = LabelEncoder().fit_transform(y)
    else:
        X = pd.get_dummies(X)
    
    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Models to evaluate
    if problem_type == 'classification':
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "k-NN": KNeighborsClassifier()
        }
        metric = 'accuracy'
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR(),
            "k-NN Regressor": KNeighborsRegressor()
        }
        metric = 'neg_mean_squared_error'

    # Evaluate models
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric)
        results[name] = np.mean(scores)
    
    # Select the best model
    best_model = max(results, key=results.get) if problem_type == 'classification' else min(results, key=results.get)
    print(f"Best model: {best_model} with {metric}: {results[best_model]}")
    
    return best_model, models[best_model]

# Example usage
# Replace 'data.csv' and 'target_column' with your dataset and target column name
data = pd.read_csv("data.csv")
best_model_name, best_model = recommend_model(data, target_column="target")
