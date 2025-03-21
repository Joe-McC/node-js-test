@echo off
echo Starting ML Workflow Tool...

REM Use full path to Python in conda environment and clear Python environment variables
set PYTHONPATH=
set PYTHONHOME=
set PYTHON_PATH=C:\Users\mccoo\Miniconda3\envs\ml-app-env\python.exe

if not exist "%PYTHON_PATH%" (
    echo ERROR: Could not find Python at %PYTHON_PATH%
    echo Please run setup.bat to create the conda environment.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_PATH%

REM Ensure Flask is installed with the correct version
echo Installing/verifying Flask installation...
"%PYTHON_PATH%" -m pip install flask==2.3.2 flask-cors==4.0.0 --quiet

echo.
echo Starting backend server...
start cmd /k "cd src\backend && "%PYTHON_PATH%" app.py"

echo Starting frontend server...
start cmd /k "npm start"

echo.
echo ML Workflow Tool started. Please wait for the application to open in your browser.
echo If the application doesn't open automatically, navigate to http://localhost:3000 