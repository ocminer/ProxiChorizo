@echo off
setlocal
rem Enter your pool token below, then double-click this file. proxima.key from the pool in this directory is used for VRF jobs (until slot 196000); hash jobs (from slot 197000) need no key.
set "TOKEN=YOUR_TOKEN"
set "WORKER=windows"
set "POOL=stratum+tcp://proxi.suprnova.cc:3355"
if "%TOKEN%"=="YOUR_TOKEN" (
  echo Edit start.cmd and set TOKEN to the token from your pool registration first.
  pause
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Token "%TOKEN%" -Worker "%WORKER%" -Pool "%POOL%" -KeyFile "%~dp0proxima.key"
pause
