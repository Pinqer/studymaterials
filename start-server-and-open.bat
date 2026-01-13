@echo off
echo ========================================
echo  Starting Intro DS Exam Prep Server
echo ========================================
echo.
echo Server will start on: http://localhost:8000
echo Opening browser...
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Open the browser
start http://localhost:8000

REM Start the Python HTTP server
python -m http.server 8000

pause
