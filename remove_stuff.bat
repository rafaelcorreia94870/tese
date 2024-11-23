@echo off
:: Specify the target folder here, or use the current directory.
set "folder=C:\5ano\cuda"

:: Delete .exe, .exp, and .lib files
del /f /q "%folder%\*.exe"
del /f /q "%folder%\*.exp"
del /f /q "%folder%\*.lib"

:: Confirmation message
echo Files with .exe, .exp, and .lib extensions have been deleted from %folder%.

pause
