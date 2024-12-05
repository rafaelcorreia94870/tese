# Compiler and flags
NVCC = nvcc
NVCCFLAGS = --extended-lambda -std=c++17 -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
LDFLAGS = -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64" -lcudart

# Files
TARGET = program
SRC = main.cpp cuda_map.cu

# Rules
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	del $(TARGET)
