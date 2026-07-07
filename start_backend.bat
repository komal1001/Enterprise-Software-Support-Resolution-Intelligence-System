@echo off
cd /d "%~dp0"
C:\Users\shubh\miniforge3\envs\python_env\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
