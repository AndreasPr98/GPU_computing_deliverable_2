# Toolchains
CXX = g++
NVCC = nvcc
CXXFLAGS = -Iinclude -O3 -fopenmp -g -std=c++17
DEPFLAGS = -MMD -MP
NVCCFLAGS = -Iinclude -O3 -g -std=c++17 -Xcompiler -fopenmp \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90

# 1. Folders
BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
INC_FOLDER := include
BATCH_OUT_FOLDER := program_output

# 2. Main Executable Info
MAIN_NAME = SpMV
MAIN_BIN = $(MAIN_NAME)
# Source file renamed to match the new executable name
MAIN_SRC = SpMV.cu

# 3. Math library
LIBS = -lm -Xcompiler -fopenmp -lcusparse

CPP_OBJECTS = $(OBJ_FOLDER)/time_and_path_management.o $(OBJ_FOLDER)/utils.o $(OBJ_FOLDER)/parser.o $(OBJ_FOLDER)/CPU_kernels.o
CU_OBJECTS = $(OBJ_FOLDER)/GPU_kernels.o
OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)
DEPS = $(OBJECTS:.o=.d)

all: $(BIN_FOLDER)/$(MAIN_BIN)

# Standard rule for compiling .cpp to .o
$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.cpp
	@mkdir -p $(OBJ_FOLDER)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# Rule for compiling CUDA source files in src/
$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.cu
	@mkdir -p $(OBJ_FOLDER)
	$(NVCC) $(NVCCFLAGS) $(DEPFLAGS) -c $< -o $@

# Build final executable
$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER) $(BATCH_OUT_FOLDER) 
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) $(OBJECTS) -o $@ $(LIBS)

clean:
	rm -rf $(OBJ_FOLDER) $(BIN_FOLDER)

-include $(DEPS)
