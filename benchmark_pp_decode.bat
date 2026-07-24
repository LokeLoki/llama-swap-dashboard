@echo off
title llama-server Benchmark - PP & Decode
cd /d "%~dp0"

echo Starting benchmark...
echo.

python benchmark_pp_decode.py

echo.
echo ========================================
echo Benchmark finished.
echo ========================================
pause
