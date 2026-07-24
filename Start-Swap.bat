@echo off
title Llama-Swap Proxy Monitor
cd /d "%~dp0"
.\llama-swap.exe -config config.yaml -listen :8081
echo.
echo [SERVER HAS CRASHED OR STOPPED]
pause