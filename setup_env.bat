@echo off
REM Windows setup.  Usage:  setup_env.bat
REM Change CUDA_TAG for older drivers (cu124, cu121).
if "%CUDA_TAG%"=="" set CUDA_TAG=cu128

python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo ^>^>^> Installing PyTorch (CUDA wheel tag: %CUDA_TAG%)
pip install torch --index-url https://download.pytorch.org/whl/%CUDA_TAG%

echo ^>^>^> Installing remaining libraries
pip install -r requirements.txt

echo ^>^>^> Verifying GPU
python scripts\00_verify_env.py
