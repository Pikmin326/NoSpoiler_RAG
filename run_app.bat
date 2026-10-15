@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/2] Preparing novel (entity dictionary + summaries, resumes if already done)...
".conda\python.exe" -m spoiler_rag.prepare
echo.
echo [2/2] Starting app... (close this window to stop)
".conda\python.exe" -m streamlit run app.py
pause
