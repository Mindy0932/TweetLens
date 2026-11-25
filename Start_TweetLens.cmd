@echo off
set "HF_HOME=%~dp0..\.hf-cache"
set "PYTHONIOENCODING=utf-8"
set "TWEETLENS_PYTHON=%~dp0..\.venv-tweetlens\Scripts\python.exe"
if not exist "%TWEETLENS_PYTHON%" set "TWEETLENS_PYTHON=python"
echo Open http://127.0.0.1:8765/TweetLens_Showcase.html after the server starts.
"%TWEETLENS_PYTHON%" "%~dp0serve_showcase.py"
pause
