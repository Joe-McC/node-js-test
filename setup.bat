@echo off
echo Setting up AI Model Verification Tool...

echo Installing frontend dependencies...
call npm install

echo Creating Python virtual environment...
cd src\backend
python -m venv venv
call venv\Scripts\activate

echo Installing Python dependencies...
pip install -r requirements.txt

echo Setup complete!
echo Run 'start.bat' to start the application

cd ..\..
pause 