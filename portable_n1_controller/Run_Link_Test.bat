@echo off
rem Over-the-air command-link test: sends numbered packets to the ESP32 and
rem shows a live line per packet (ACK/LOST, round-trip ms, WiFi RSSI).
rem Usage:  Run_Link_Test.bat [ip:port] [count]
cd /d "%~dp0"
set "ESP=%~1"
if "%ESP%"=="" set "ESP=192.168.86.199:8888"
set "COUNT=%~2"
if "%COUNT%"=="" set "COUNT=200"
set "PY=python"
where python >nul 2>nul || set "PY=G:\CondaEnvs\env_isaaclab\python.exe"
"%PY%" tools\link_test.py --esp %ESP% --count %COUNT%
echo.
pause
