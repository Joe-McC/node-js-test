@echo off
echo Starting ML Workflow Tool...

echo Checking Python environment...
if not exist "src\backend\venv" (
    echo Creating virtual environment...
    cd src\backend
    python -m venv venv
    cd ..\..
)

echo Installing required packages...
call src\backend\venv\Scripts\activate.bat
pip install flask flask-cors pandas numpy scikit-learn matplotlib
echo Python environment ready.

echo Starting backend server...
start cmd /k "cd src\backend && call venv\Scripts\activate.bat && python app.py"

echo Starting frontend server...
start cmd /k "npm start"

echo ML Workflow Tool started. Please wait for the application to open in your browser.
echo If the application doesn't open automatically, navigate to http://localhost:3000 