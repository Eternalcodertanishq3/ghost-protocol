@echo off
REM ============================================================
REM  GHOST PROTOCOL - HOSPITAL LAPTOP STARTUP
REM  Copy this folder to each hospital laptop and run this script
REM ============================================================

echo.
echo  ============================================================
echo    GHOST PROTOCOL - HOSPITAL AGENT SETUP
echo  ============================================================
echo.

set /p HOSPITAL_ID="Enter Hospital Name (e.g., AIIMS_Delhi): "
set /p SERVER_IP="Enter Central Server IP (e.g., 192.168.1.100): "

echo.
echo  Starting Ghost Agent for %HOSPITAL_ID%...
echo  Connecting to server at %SERVER_IP%:8000
echo.

cd /d "%~dp0"
python hospital_agent.py --hospital %HOSPITAL_ID% --server %SERVER_IP%:8000 --rounds 10 --delay 10

pause
