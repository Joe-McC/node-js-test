@echo off
echo Setting up AI Model Verification Tool...

echo Installing frontend dependencies...
call npm install

echo Do you want to use Conda (recommended) or venv?
echo 1. Conda (recommended)
echo 2. Python venv
set /p env_choice="Enter choice (1 or 2): "

if "%env_choice%"=="1" (
    echo Setting up Conda environment...
    
    REM Check if conda exists in the user profile
    if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" (
        set "CONDA_EXE=%USERPROFILE%\Miniconda3\Scripts\conda.exe"
    ) else if exist "%USERPROFILE%\Anaconda3\Scripts\conda.exe" (
        set "CONDA_EXE=%USERPROFILE%\Anaconda3\Scripts\conda.exe"
    ) else (
        echo Conda installation not found.
        echo Please install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
        pause
        exit /b 1
    )

    echo Found Conda at: %CONDA_EXE%
    echo.
    echo Initializing Conda...
    call "%CONDA_EXE%" init cmd.exe

    echo.
    echo Creating or updating the ml-app-env environment...
    call "%CONDA_EXE%" env create -f environment.yml || call "%CONDA_EXE%" env update --file environment.yml --prune

    echo.
    echo Conda setup completed successfully!
    echo IMPORTANT: You will need to close and reopen your command prompt before using conda.
) else (
    echo Creating Python virtual environment...
    cd src\backend
    python -m venv venv
    call venv\Scripts\activate

    echo Installing Python dependencies...
    pip install -r requirements.txt
    
    cd ..\..
)

echo Setup complete!
echo Run 'start.bat' to start the application
echo Note: If you chose Conda, please close and reopen your command prompt first.

pause 