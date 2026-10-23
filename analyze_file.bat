@echo off
setlocal
cd /d "%~dp0"
set /p INPUT=Paste the full path to the EXE or DLL (without surrounding quotes): 
if not exist "%INPUT%" (
  echo ERROR: File not found: %INPUT%
  pause
  exit /b 2
)
py .\independent_decompiler_v6.py "%INPUT%" --out "analysis_output" --fast-output
if errorlevel 1 (
  echo Analysis failed. Check the error above and confirm dependencies are installed.
  pause
  exit /b 1
)
echo.
echo Results saved to: %CD%\analysis_output
explorer "%CD%\analysis_output"
pause
