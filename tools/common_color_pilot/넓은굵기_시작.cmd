@echo off
chcp 65001 >nul
where py >nul 2>nul
if errorlevel 1 (python "%~dp0wide_pilot.py") else (py -3 "%~dp0wide_pilot.py")
if errorlevel 1 pause
