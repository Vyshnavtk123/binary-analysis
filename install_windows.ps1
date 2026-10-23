$ErrorActionPreference = "Stop"
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
Write-Host "Dependencies installed. Try: py .\independent_decompiler.py .\sample.exe --out .\analysis_output --fast-output"
