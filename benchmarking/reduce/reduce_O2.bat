@echo off
set MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64

nvcc -G -g ^
     -O2 ^
     -std=c++20 ^
     -ccbin "%MSVC_PATH%" ^
     --extended-lambda --expt-relaxed-constexpr ^
     -o reduceO2 reduce.cu
