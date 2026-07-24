@echo off
title CUDA RPC Worker 50053
echo ========================================================
echo  ISOLATING ENVIRONMENT TO 3070 Ti AND 5060 Ti
echo  Hardware Lock: GPU-01c55f82-15b2-6238-f7ff-e24f3101a06b
echo                 GPU-e3a855d6-81c1-1bc9-5b88-64d0f7bb9868
echo ========================================================

set CUDA_VISIBLE_DEVICES=GPU-01c55f82-15b2-6238-f7ff-e24f3101a06b,GPU-e3a855d6-81c1-1bc9-5b88-64d0f7bb9868

cd C:\llama-cuda\

rpc-server.exe --host 127.0.0.1 --port 50053
pause