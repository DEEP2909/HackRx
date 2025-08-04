@echo off
echo Setting up LLM Query Retrieval System...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Copy environment file
echo Setting up environment...
if not exist .env (
    copy .env.example .env
    echo Please edit .env file with your API keys!
)

echo.
echo Setup complete! To run the application:
echo 1. Edit .env file with your API keys
echo 2. Run: venv\Scripts\activate
echo 3. Run: python -m uvicorn app.main:app --reload
echo.
pause
