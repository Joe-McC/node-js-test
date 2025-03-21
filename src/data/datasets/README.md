# Example Datasets

This directory contains example datasets for testing and demonstrating the ML Workflow Builder application. These datasets can be used with the various nodes in the workflow to build and evaluate machine learning models.

## Available Datasets

### 1. Iris Dataset (`iris.csv`)
- **Description**: The famous Iris flower dataset
- **Features**: Sepal length, sepal width, petal length, petal width
- **Target**: Species (setosa, versicolor, virginica)
- **Type**: Classification (Multi-class)
- **Size**: 150 samples
- **Use Case**: Good for basic classification tasks

### 2. Boston Housing (`boston_housing.csv`)
- **Description**: Housing data for Boston suburbs
- **Features**: Crime rate, zoning, industry, Charles River dummy variable, etc.
- **Target**: MEDV (Median value of owner-occupied homes in $1000s)
- **Type**: Regression
- **Size**: 506 samples
- **Use Case**: Regression problems, housing price prediction

### 3. Customer Churn (`customer_churn.csv`)
- **Description**: Telecom customer churn prediction dataset
- **Features**: Demographics, service usage, payment information
- **Target**: Churn (Yes/No)
- **Type**: Classification (Binary)
- **Size**: 30 samples (sample version)
- **Use Case**: Customer retention, binary classification

### 4. Wine Quality (`wine_quality.csv`)
- **Description**: Wine quality based on physicochemical tests
- **Features**: Acidity, sugar, pH, alcohol, etc.
- **Target**: Quality (score between 0 and 10)
- **Type**: Regression/Classification
- **Size**: 30 samples (sample version)
- **Use Case**: Quality prediction, regression or ordinal classification

### 5. Diabetes (`diabetes.csv`)
- **Description**: Diabetes prediction dataset
- **Features**: Pregnancies, glucose, blood pressure, insulin, BMI, etc.
- **Target**: Outcome (1 = has diabetes, 0 = no diabetes)
- **Type**: Classification (Binary)
- **Size**: 50 samples (sample version)
- **Use Case**: Disease prediction, healthcare analytics

## Usage

These datasets can be loaded into the ML Workflow Builder using the DataPrep node. Select the appropriate dataset based on your machine learning task.

For classification problems, consider:
- Iris (multi-class)
- Customer Churn (binary)
- Diabetes (binary)

For regression problems, consider:
- Boston Housing
- Wine Quality

## Data Splitting

When using these datasets, remember to:
1. Split the data into training and testing sets
2. Consider preprocessing steps like scaling or normalization
3. Handle categorical variables appropriately

The ML Workflow Builder provides nodes to help with these tasks. 