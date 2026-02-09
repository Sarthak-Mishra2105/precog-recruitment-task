@echo off
REM Build script for Colored MNIST CV Task Report
REM ==============================================
REM
REM Usage: build.bat
REM
REM Requires: pdflatex and bibtex in PATH (e.g., from MiKTeX or TeX Live)

echo Building report...

cd /d "%~dp0"

echo [1/4] First pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: pdflatex failed on first pass
    pause
    exit /b 1
)

echo [2/4] Running bibtex...
bibtex main
if errorlevel 1 (
    echo WARNING: bibtex had issues (may be okay if no citations)
)

echo [3/4] Second pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: pdflatex failed on second pass
    pause
    exit /b 1
)

echo [4/4] Third pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: pdflatex failed on third pass
    pause
    exit /b 1
)

echo Renaming to report.pdf...
if exist main.pdf (
    move /Y main.pdf report.pdf
    echo.
    echo SUCCESS: report.pdf created!
) else (
    echo ERROR: main.pdf not found
    pause
    exit /b 1
)

echo.
echo Build complete. Output: report.pdf
