@echo off
cd /d "%~dp0"
echo ============================================
echo       VENOMGUARD AI - LOCALHOST
echo ============================================
echo.
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
pause
