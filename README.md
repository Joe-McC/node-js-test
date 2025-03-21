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

### Backend Setup

You can choose between a standard Python virtual environment (venv) or a Conda environment:

1. Run the setup script and choose your preferred environment:
   ```
   setup.bat
   ```
   This will install all necessary dependencies for either environment type.

2. Start the application:
   ```
   start.bat
   ```
   The script will automatically detect and use the available environment.

**Note for Conda users:** Make sure [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) is installed before running setup.bat.

## Known Issues and Workarounds

### Conda Environment Import Issues

There's a known issue with conda environments where module imports (such as Flask) may fail even when the packages are correctly installed in the environment. This can happen due to:

1. Environment variable conflicts
2. Multiple Python installations or conda environments with the same name
3. PATH resolution problems with conda activation

The current workaround in `start.bat` uses direct paths to the Python executable in the conda environment, bypassing conda activation mechanisms:

```batch
# Use full path to Python in conda environment
set PYTHONPATH=
set PYTHONHOME=
set PYTHON_PATH=C:\Users\<username>\Miniconda3\envs\ml-app-env\python.exe

# Directly run with that Python interpreter
"%PYTHON_PATH%" app.py
```

Future updates should consider more robust environment detection and potentially use a more cross-platform approach.

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
