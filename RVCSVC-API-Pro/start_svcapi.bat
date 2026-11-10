@echo off
chcp 65001 >nul
title SVC API WebUI
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime;%SCRIPT_DIR%\runtime\Scripts;%PATH%"
rem 绕过系统代理，防止 Gradio 启动自检（localhost:7866）被代理截胡返回 502
set "NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1"
set "no_proxy=localhost,127.0.0.1,0.0.0.0,::1"
runtime\python.exe app_svc.py
pause
