//#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h> //clock_gettime

void Sum (float* a, float* b, float* c, unsigned int n){
	for(int i = 0; i < n; i++){
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char* argv[])
{
    int n = atoi(argv[1]);
	int BLOCK_SIZE = atoi(argv[2]);
	int GRID_SIZE = atoi(argv[3]);
	int count_iteration = atoi(argv[4]);
	int size_byte = n * sizeof(float);

	if (count_iteration <= 0)
		count_iteration = 5;

    //Определяем размер грида и блоков
	if (BLOCK_SIZE <= 0)
		BLOCK_SIZE = 1;
	if (GRID_SIZE <= 0)
 		GRID_SIZE = 1;
	
	float FullTimeCPU = 0;
	float FullTimeGPU = 0;

	for (int iteration = 0 ; iteration < count_iteration ; iteration++)
{
// Выделение памяти на хосте-CPU
    float* a = (float*)calloc(n, sizeof(float));
    float* b = (float*)calloc(n, sizeof(float));
    float* cGpu = (float*)calloc(n, sizeof(float));
	float* cCpu = (float*)calloc(n, sizeof(float));

    for (int i = 0; i < n; i++) 
	{
        	a[i] = (float)(rand() % 100) / (float)(rand() % 100);
			b[i] = (float)(rand() % 100) / (float)(rand() % 100);
    }

    // Выделение памяти на устройстве
    float* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, size_byte);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, size_byte);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, size_byte);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cuerr = cudaEventCreate(&start);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaEventCreate(&stop);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Копирование данных с хоста на девайс
    cuerr = cudaMemcpy(adev, a, size_byte, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(bdev, b, size_byte, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Установка точки старта
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    //Запуск ядра
    addKernel <<< GRID_SIZE, BLOCK_SIZE >>> (cdev, adev, bdev, n);

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    // Синхронизация устройств
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Копирование результата на хост
    cuerr = cudaMemcpy(cGpu, cdev, size_byte, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

	struct timespec mt1, mt2; 
    //Переменная для расчета дельты времени
    float cpuTime;

	clock_gettime(CLOCK_REALTIME, &mt1);
    Sum(a, b, cCpu, n);
    clock_gettime(CLOCK_REALTIME, &mt2);
    
    // Расчет времени
	cpuTime=float(1000000000*(mt2.tv_sec - mt1.tv_sec)+(mt2.tv_nsec - mt1.tv_nsec))/1000000;
	FullTimeCPU+=cpuTime;
	
	printf("CPU time: %f ms \n", cpuTime);
    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    printf("GPU time %s: %.9f ms\n", "kernel", gpuTime);
	FullTimeGPU+=(float)gpuTime;
	printf("\n");

	float error = 0;
	for(int i = 0; i< i; i++) {
		cCpu[i] -= cGpu[i];
		if(abs(cCpu[i]) > error) 
			error = abs(cCpu[i]);
	}
	printf("Error: %.9f \n", error);
	printf("SpeedUp: %.9f \n", cpuTime/gpuTime);
	printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(cGpu);
	free(cCpu);
}
	printf("Average CPU time: %f ms \n", FullTimeCPU/count_iteration);
	printf("Average GPU time: %f ms \n", FullTimeGPU/count_iteration);
    
    return 0;
}
