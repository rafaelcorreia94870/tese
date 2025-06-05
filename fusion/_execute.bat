
@echo off
REM List all your batch files here, one per line
setlocal

set BATCH_FILES=basic_fused_gpu.bat experimental_fusion.bat fused_cpu_vec.bat fused_gpu_vec.bat fused_vector.bat

for %%F in (%BATCH_FILES%) do (
    echo Running %%F...
    call "%%F"
    echo.

    REM Get the base name (remove .bat extension)
    set "BASENAME=%%~nF"
    call :runexe "%%BASENAME%%"
)

echo All scripts and executables executed.
pause
exit /b

:runexe
setlocal
set "EXE=%~1.exe"
if exist "%EXE%" (
    echo Running %EXE%...
    "%EXE%"
    echo.
) else (
    echo %EXE% not found.
)
endlocal
exit /b