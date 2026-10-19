@echo off
setlocal
rem Enter your pool token below (proxima.key from the pool must be in this directory), then double-click this file.
set "TOKEN=YOUR_TOKEN"
set "WORKER=windows"
set "POOL=http://proxi.suprnova.cc:8100"
if "%TOKEN%"=="YOUR_TOKEN" (
  echo Edit start.cmd and set TOKEN to the token from your pool registration first.
  pause
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Token "%TOKEN%" -Worker "%WORKER%" -Pool "%POOL%" -KeyFile "%~dp0proxima.key"
pause
