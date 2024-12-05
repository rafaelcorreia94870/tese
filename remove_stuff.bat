@echo off
:: Specify the target folder here, or use the current directory.
set "folder=C:\5ano\tese"

:: Delete .exe, .exp, and .lib files
del /f /q "%folder%\*.exe"
del /f /q "%folder%\*.exp"
del /f /q "%folder%\*.lib"
del /f /q "%folder%\*.o"
del /f /q "%folder%\*.obj"



:: Confirmation message
echo Files with .exe, .exp, .lib, .o and .obj extensions have been deleted from %folder%.

pause
