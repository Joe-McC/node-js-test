# ML Workflow Tool

A machine learning workflow visualization and management tool built with React and Flask.

## Features

- **Model Evaluation**: Compare reference and current data distributions, detect data drift, and analyze target drift using Evidently AI.
- **Model Testing**: Analyze data characteristics, feature distributions, and correlations with target variables.
- **Parameter Tuning**: Optimize model hyperparameters with grid search and visualize results.

## Setup

### Frontend (React)

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

### Backend (Flask)

1. Install Python dependencies:
   ```
   cd src/backend
   pip install flask flask-cors pandas numpy scikit-learn matplotlib plotly evidently
   ```

2. Start the Flask server:
   ```
   python app.py
   ```

## Usage

The application provides a visual interface for common machine learning workflows:

1. **Model Evaluation**: Upload reference and current datasets to analyze data quality and drift metrics with Evidently AI.
2. **Model Testing**: Evaluate model performance on test data with various metrics and visualizations.
3. **Parameter Tuning**: Find optimal hyperparameters for your models with grid search.

## Backend API

The Flask backend provides the following API endpoints:

- `/api/model-eval`: Evaluates data quality and drift between reference and current datasets using Evidently
- `/api/model-test`: Analyzes test data characteristics and feature importance with Evidently reporting
- `/api/parameter-tuning`: Performs hyperparameter optimization with grid search

## Technologies Used

- **Frontend**: React, React Flow, CSS
- **Backend**: Flask, Pandas, NumPy, scikit-learn, Matplotlib, Plotly
- **ML Monitoring**: Evidently AI (version 0.14.4)
