@echo off
echo Starting Llamafile server for DataWhale AI...
echo Model: models\qwen2.5-3b-instruct-q4_k_m.gguf
echo Endpoint: http://127.0.0.1:8080/v1
echo Press Ctrl+C to stop.
echo.

llamafile-0.10.6.exe -m models\qwen2.5-3b-instruct-q4_k_m.gguf --port 8080 --host 127.0.0.1
pause
