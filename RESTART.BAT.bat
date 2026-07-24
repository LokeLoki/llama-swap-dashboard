@echo off
echo Killing all AI processes (Safe RAM clear, NO files deleted)...
taskkill /F /IM llama-swap.exe /T >nul 2>&1
taskkill /F /IM llama-server.exe /T >nul 2>&1
taskkill /F /IM rpc-server.exe /T >nul 2>&1
taskkill /F /IM ollama_llama_server.exe /T >nul 2>&1
taskkill /F /IM ollama.exe /T >nul 2>&1
taskkill /F /IM hermes.exe /T >nul 2>&1
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM pythonw.exe /T >nul 2>&1
taskkill /F /IM bash.exe /T >nul 2>&1

echo Processes killed. Waiting for VRAM to clear...
timeout /t 2

echo Starting Dedicated Hermes Manager (5060 Ti + Vision) on Port 8089...
cd /d "C:\Users\khang\Desktop\POOL HOST X2\models\ORNITH-1.0-9B-MTP-GGUF"
start "Hermes Manager" "Start-Hermes-Manager.bat"


echo Waiting for Workers to stabilize...
timeout /t 3

echo Starting Hermes Gateway...
cd /d "C:\Users\khang\Desktop\POOL HOST X2"
powershell.exe -ExecutionPolicy Bypass -File "start_hermes.ps1"

echo Launching Coordinator Proxy...
cd /d "C:\Users\khang\Desktop\POOL HOST X2"
start "Llama Swap" "Start-Swap.bat"

@echo off
echo Starting ComfyUI on port 8188...
echo Access at: http://localhost:8188
set PYTHONPATH=
cd /d C:\Users\khang\Desktop\ComfyUI
call venv\Scripts\activate
python main.py --listen 0.0.0.0 --port 8188

echo All systems initialized.
exit