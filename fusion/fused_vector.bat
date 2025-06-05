@echo off
set MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64

nvcc -std=c++20 ^
    --extended-lambda  --expt-relaxed-constexpr^
    -ccbin "%MSVC_PATH%" ^
    -o fused_vector fused_vector.cu ^
    -g -G -arch=sm_86
