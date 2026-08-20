@echo off
chcp 949 >nul
cd /d "%~dp0.."

set PY=python
where python >nul 2>&1 || set PY=py

echo 필요한 패키지를 설치합니다...
%PY% -m pip install numpy pygame matplotlib pillow
echo.
echo 전체 점검을 실행합니다...
%PY% -m colorvision.selftest
echo.
pause
