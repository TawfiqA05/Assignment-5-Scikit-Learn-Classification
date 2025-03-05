# Assignment-5-Scikit-Learn-Classification

## Purpose
This project applies machine learning classification using Python and Scikit-Learn on the built-in breast cancer dataset. The objective is to build, train, and evaluate three classification models—Logistic Regression, Decision Tree, and Support Vector Machine (SVM)—and to determine the best performing model using standard evaluation metrics.

## Project Structure
- **main.py**: Contains the Python script for data loading, preprocessing, model training, evaluation, and output.
- **ReadMe.md**: Provides an overview of the project, design choices, and explanations of key components.

## Design and Implementation
The project is implemented as a single procedural script. The central function `main()` performs the following steps:
- **Data Preparation**: Loads the dataset and splits it into training and testing sets.
- **Model Building**: Instantiates and trains three different classification models.
- **Model Evaluation**: Computes performance metrics (accuracy, precision, recall, and F1 score) for each model and prints the results.

### Variables and Methods
- **Variables**: All variables are named using snake_case (e.g., `x_data`, `y_data`, `log_reg_model`).
- **Function**: 
  - `main()`: Orchestrates data processing, model training, evaluation, and result display.

### Limitations
- The script does not perform hyperparameter tuning, which might improve model performance.
- No additional feature engineering is applied beyond the default dataset configuration.
