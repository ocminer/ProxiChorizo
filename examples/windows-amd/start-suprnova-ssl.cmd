@echo off
rem ProxiChorizo - mine Proxima (PROX) on suprnova.cc. AMD GPUs (OpenCL), TLS-encrypted.
rem 1. Register at https://proxi.suprnova.cc/StartMining.html with your PROX address; copy the TOKEN.
rem 2. Set TOKEN and WORKER below, save, double-click it (works from the examples folder or next to proxichorizo-cl.exe).
set "TOKEN=YOUR_POOL_TOKEN"
set "WORKER=rig1"
set "POOL=stratum+ssl://proxi.suprnova.cc:3356"
if "%TOKEN%"=="YOUR_POOL_TOKEN" ( echo Edit %~nx0 and set TOKEN first. & pause & exit /b 2 )
if exist "%~dp0proxichorizo-cl.exe" (cd /d "%~dp0") else (cd /d "%~dp0..")
rem Optional: -d 0,1 = only these GPUs, --max-temp 85 = per-GPU pause limit, --api-bind 127.0.0.1:4078 = stats
proxichorizo-cl.exe --proxi %POOL% --token %TOKEN% --name %WORKER%
pause
