@echo off
chcp 949 >nul
cd /d "%~dp0.."

set PY=python
where python >nul 2>&1 || set PY=py

echo ============================================
echo   결과 분석 (가장 최근 파일)
echo ============================================
echo.

for /f "delims=" %%f in ('dir /b /o-d "data\colorvision\*_trials.csv" 2^>nul') do (
    set LATEST=data\colorvision\%%f
    goto :found
)
echo data\colorvision 폴더에 결과 파일이 없습니다. 먼저 테스트를 실행하세요.
pause
exit /b 1

:found
echo 분석할 파일: %LATEST%
echo.
%PY% -m colorvision.estimate "%LATEST%" --plot --json --bootstrap 150

echo.
pause
