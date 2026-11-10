@echo off
chcp 65001 >nul
title RVC API WebUI
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime;%SCRIPT_DIR%\runtime\Scripts;%PATH%"
set "NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1"
set "no_proxy=localhost,127.0.0.1,0.0.0.0,::1"
set "PYTHONIOENCODING=utf-8"
runtime\python.exe -X utf8 app_rvc.py
pause