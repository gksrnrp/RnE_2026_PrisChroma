@echo off
chcp 949 >nul
cd /d "%~dp0.."

set PY=python
where python >nul 2>&1 || set PY=py

echo ============================================
echo   색쌍 구별 테스트
echo ============================================
echo.
set /p PID=피험자 번호를 입력하세요 (예: P01):
if "%PID%"=="" set PID=anon
echo.
echo 전체화면으로 시작합니다. 약 10분 걸립니다.
echo   - 마우스 클릭 또는 숫자키 1 2 3 4
echo   - 모르겠어도 반드시 하나를 고르세요
echo   - 중단하려면 ESC
echo.
pause

%PY% -m colorvision.experiment --participant %PID%

echo.
echo 끝났습니다. data\colorvision 폴더에 저장되었습니다.
pause
