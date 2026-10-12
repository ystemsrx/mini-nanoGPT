@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto :error

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

".venv\Scripts\python.exe" app.py
if errorlevel 1 goto :error
pause
exit /b 0

:error
echo Failed to set up or run Mini NanoGPT. See the error above.
pause
exit /b 1
