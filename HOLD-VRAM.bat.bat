@echo off
title VRAM HOLDER - 5060 Ti BOUND
echo ========================================================
echo  ALLOCATING ~1.2GB VRAM ON 5060 Ti (Vulkan0)
echo  Press CTRL+C or Close Window to release this VRAM block
echo ========================================================

set GGML_VK_VISIBLE_DEVICES=0
set CUDA_VISIBLE_DEVICES=-1

cd C:\llama-vulkan\

llama-server.exe --port 9999 -m "C:\Users\khang\Desktop\POOL HOST X2\models\QWEN3.6-27B-ABL-Q4_K_XL\Huihui-Qwen3.6-27B-abliterated-UD-Q4_K_XL-MTP.gguf" -c 1024 -ngl 4 -b 512 -ub 512 --split-mode none

echo.
echo Warning: llama-server closed unexpectedly. 
pause