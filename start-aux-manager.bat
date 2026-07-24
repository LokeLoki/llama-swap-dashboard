@echo off
color 0D
echo ========================================================
echo   ISOLATING ENVIRONMENT TO 5060 Ti (Auxiliary)
echo   Hardware Lock: GPU-e3a855d6-81c1-1bc9-5b88-64d0f7bb9868
echo ========================================================

:: 1. The Nvidia Lock
set CUDA_VISIBLE_DEVICES=GPU-e3a855d6-81c1-1bc9-5b88-64d0f7bb9868

:: 2. The Blinders
set OLLAMA_VULKAN=0
set GGML_VK_VISIBLE_DEVICES=-1
set ROCR_VISIBLE_DEVICES=-1

:: 3. The Port Lock
set OLLAMA_HOST=127.0.0.1:8090

echo Starting dedicated Auxiliary Manager on Port 8090...
ollama serve

pause