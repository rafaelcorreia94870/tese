@echo off
setlocal

set EXECUTABLES=reduce.exe reduceO2.exe reduceO3.exe

for %%E in (%EXECUTABLES%) do (
    if exist "%%E" (
        echo Running %%E...
        call "%%E"
        echo.
    ) else (
        echo %%E not found.
        echo.
    )
)

echo All executables executed.
pause
