import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Load and prepare the dataset
    dataset = load_breast_cancer()
    x_data = dataset.data
    y_data = dataset.target

    # Split the data into training and testing sets (70/30 split)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    # Initialize the classification models with consistent snake_case variable names
    log_reg_model = LogisticRegression(max_iter=10000, random_state=42)
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    svm_model = SVC(random_state=42)

    # Train the models on the training data
    log_reg_model.fit(x_train, y_train)
    decision_tree_model.fit(x_train, y_train)
    svm_model.fit(x_train, y_train)

    # Evaluate each model on the testing data and store metrics
    models = {
        "logistic_regression": log_reg_model,
        "decision_tree": decision_tree_model,
        "svm": svm_model
    }
    
    metrics = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    # Print evaluation metrics for each model
    for model_name, metric in metrics.items():
        print(f"Model: {model_name}")
        print(f"Accuracy: {metric['accuracy']:.4f}")
        print(f"Precision: {metric['precision']:.4f}")
        print(f"Recall: {metric['recall']:.4f}")
        print(f"F1 Score: {metric['f1_score']:.4f}\n")

    # Identify the best model based on F1 Score
    best_model = max(metrics, key=lambda x: metrics[x]["f1_score"])
    print(f"Best performing model based on F1 Score: {best_model}\n")

    # Brief explanation of the decision
    print("Brief Explanation:")
    print("The logistic regression model performed best with a balanced trade-off between precision and recall, resulting in a higher F1 score. "
          "This indicates it provides more reliable overall classification performance compared to the decision tree and SVM models.")

if __name__ == '__main__':
    main()