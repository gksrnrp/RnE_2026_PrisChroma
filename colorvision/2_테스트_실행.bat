@echo off
chcp 949 >nul
cd /d "%~dp0.."

set PY=python
where python >nul 2>&1 || set PY=py

echo ============================================
echo   색쌍 구별 테스트
echo ============================================
echo.
set /p PID=피험자 번호 (예: P05):
if "%PID%"=="" set PID=anon
echo.
echo 아래 두 값은 자극이 눈에 얼마나 크게 보였는지 기록하는 데 쓰입니다.
echo 대략이라도 넣어야 나중에 다른 화면 자료와 비교할 수 있습니다.
echo.
set /p DIST=화면까지의 거리 cm (엔터=50):
if "%DIST%"=="" set DIST=50
set /p DIAG=모니터 대각선 인치 (엔터=15.6):
if "%DIAG%"=="" set DIAG=15.6
echo.
echo 전체화면으로 시작합니다. 약 10분 걸립니다.
echo   - 마우스 클릭 또는 숫자키 1 2 3 4
echo   - 모르겠어도 반드시 하나를 고르세요
echo   - 중단하려면 ESC
echo.
pause

%PY% -m colorvision.experiment --participant %PID% --viewing-distance %DIST% --screen-diagonal %DIAG%

echo.
echo 끝났습니다. 결과가 저장되었습니다.
pause
