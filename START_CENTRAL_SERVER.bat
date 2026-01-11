@echo off
REM ============================================================
REM  GHOST PROTOCOL - CENTRAL SERVER STARTUP
REM  Run this on the main laptop that hosts the SNA
REM ============================================================

echo.
echo  ============================================================
echo    GHOST PROTOCOL - CENTRAL SERVER
echo  ============================================================
echo.

REM Get IP address
echo  Step 1: Finding your IP address...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    echo    Your IP: %%a
    echo    Tell other laptops to connect to:%%a:8000
)
echo.

REM Open firewall (requires admin)
echo  Step 2: Firewall (may require admin)...
netsh advfirewall firewall add rule name="Ghost Protocol SNA" dir=in action=allow protocol=TCP localport=8000 >nul 2>&1
echo    Port 8000 opened
echo.

REM Start backend
echo  Step 3: Starting SNA server on port 8000...
echo.
echo  ============================================================
echo    SNA SERVER RUNNING - READY FOR HOSPITAL CONNECTIONS
echo    Share your IP with other laptops!
echo    Press Ctrl+C to stop
echo  ============================================================
echo.

cd /d "%~dp0"
python -m sna.main
