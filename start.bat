@echo off
echo Starting Plant Disease Detection System...
echo.

echo [1/2] Starting FastAPI backend...
start "FastAPI Backend" cmd /k "venv\Scripts\activate && python run_api.py"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Streamlit frontend...
start "Streamlit Frontend" cmd /k "venv\Scripts\activate && streamlit run streamlit_app.py"

echo.
echo Both services starting...
echo API:      http://localhost:8000
echo Frontend: http://localhost:8501
echo API Docs: http://localhost:8000/docs
pause