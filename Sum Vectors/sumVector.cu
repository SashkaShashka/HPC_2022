#include <cstdio>
#include <iostream>
#include "assert.h"
#include <time.h>
using namespace std;

#define BLOCK_SIZE 1024
int N = 134217728;


// Если cudaError_t, который вернула cuda-функция, не cudaSuccess, то выводит обнаруженную ошибку
void err_check(cudaError_t error){
    if (error != cudaSuccess){
        cout << "Error" << endl;
        cerr << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

__global__ void reduce4(int* inData, int* outData)
{
    __shared__ int data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    data[tid] = inData[i] + inData[i+blockDim.x];      // load into shared memeory

    __syncthreads();

    for (int s = blockDim.x/2; s>0; s>>=1 )
    {
        if (tid < s)
           data[tid] += data[tid + s];

        __syncthreads();
    }

    if (tid == 0)                                       // write result of block reduction
        outData[blockIdx.x] = data[0];
}


long clp2(long x) {
 long p2=1;
 while (1) {
  if (p2>=x) return p2;
  p2<<=1;
 }
 return 0;
}


int main(int argc, char *  argv[])
{
    int nearest_power_of_2 = clp2(N);
    
    int * a = new int[nearest_power_of_2];
    int * b = new int[nearest_power_of_2];

    int n = nearest_power_of_2;
    int numBytes = n * sizeof(int);
    
    for (int i = 0; i < N; i++)
    {
        a[i] = (rand() & 0xFF) - 127; //(rand() & 0xFF) - 127;
    }
    
    double cpu_time;
    int sum = 0;
    
    clock_t start_cpu, end_cpu;
    start_cpu = clock();
    
    for (int i = 0; i < N; i++){
        sum  += a[i];
    }
    
    end_cpu = clock();
    cpu_time = 1.0f * (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("time spent executing cpu: %lf milliseconds\n",cpu_time*1000);
    
    for (int i = N; i < nearest_power_of_2; i++){
        a[i] = 0;
    }

    int* adev[2] = { NULL, NULL };
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    err_check(cudaMalloc((void**)&adev[0], numBytes));
    err_check(cudaMalloc((void**)&adev[1], numBytes));
    
    err_check(cudaMemcpy(adev[0], a, numBytes, cudaMemcpyHostToDevice));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    
    int grid_size = n / (2*BLOCK_SIZE);
    
    if(grid_size==0){
       grid_size=1; 
    }
       
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(grid_size, 1, 1);

    reduce4<<<dimGrid, dimBlock>>>(adev[0], adev[1]);

    err_check(cudaDeviceSynchronize());
    
    // n/=(2*BLOCK_SIZE);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    
    cudaMemcpy(b, adev[1], numBytes, cudaMemcpyDeviceToHost);

    for (int i = 1; i < grid_size; i++){
        b[0] += b[i];
    }
    
    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.4f milliseconds\n", gpuTime);
    printf("CPU sum %d, CUDA sum %d, N = %d\n", sum, b[0], N);

    // release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev[0]);
    cudaFree(adev[1]);

    delete a;
    delete b;

    return 0;
}