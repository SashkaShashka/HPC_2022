__global__ void mulKernel(float* c, float* a, float* b, unsigned int m, unsigned int n, unsigned int k) {
	//int count_threads = gridDim.x * blockDim.x;
	int Row = blockIdx.x * blockDim.x + threadIdx.x;
	int Col = blockIdx.y * blockDim.y + threadIdx.y;
	if ((Row < m) && (Col < k))
	{
		for (int i = 0; i < n; i++)
		{
			c[Row*k + Col] += a[Row*n + i] * b[i * k + Col];
		}
	}	
}
#include "mainGPU.h"