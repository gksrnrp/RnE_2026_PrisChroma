@echo off
chcp 65001 >nul
where py >nul 2>nul
if errorlevel 1 (python "%~dp0run_v5.py") else (py -3 "%~dp0run_v5.py")
if errorlevel 1 pause
