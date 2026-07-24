@echo off
echo Swapping to 27i (image gen config)...
curl -s http://127.0.0.1:8081/v1/chat/completions -X POST -H "Content-Type: application/json" -d "{\"model\":\"27i\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" >nul
echo Done! 27i loaded (5060 Ti inference, 4070 Ti Super free for ComfyUI).
pause
