@echo off
set MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64

nvcc -std=c++20 ^
    -O2 --maxrregcount=64 ^
    --extended-lambda  --expt-relaxed-constexpr^
    -arch=sm_86 -Xptxas=-v -lineinfo ^
    -ccbin "%MSVC_PATH%" ^
    -o fusion_benchmark fusion_benchmark.cu 