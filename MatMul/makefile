# The size of shared memory block size
NVCC = nvcc 
CFLAGS = -g -G -O0
MatMul: matmulGPU.cu mainGPU.h
	$(NVCC) $(CFLAGS) $< -o $@