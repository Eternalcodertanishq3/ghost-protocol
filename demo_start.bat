@echo off
REM Ghost Protocol - Demo Startup Script
REM Starts both backend and frontend for hackathon demo

echo ========================================
echo   Ghost Protocol v1.0 - Demo Startup
echo   DPDP-Safe Federated Learning
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "sna\main.py" (
    echo ERROR: Please run this script from the ghost-protocol directory
    exit /b 1
)

echo [1/3] Starting Backend (Secure National Aggregator)...
start "Ghost Protocol - Backend" cmd /k "python -m sna.main"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo [2/3] Starting Frontend (React Dashboard)...
start "Ghost Protocol - Frontend" cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo   Ghost Protocol is starting up!
echo ========================================
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo.
echo   Wait for both terminals to show "ready"
echo   then open http://localhost:3000 in your browser
echo.
echo ========================================
echo.
pause
