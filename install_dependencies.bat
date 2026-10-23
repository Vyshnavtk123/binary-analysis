@echo off
setlocal
cd /d "%~dp0"
echo Installing required Python packages: pefile and capstone
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check your internet connection and Python installation.
  pause
  exit /b 1
)
echo Dependencies installed successfully.
pause
