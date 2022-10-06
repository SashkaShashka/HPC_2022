#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h> 

int main(int argc, char* argv[])
{
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    printf("m = %d\n", m);
    printf("n = %d\n", n);
    printf("k = %d\n", k);

    // Выделение памяти на хосте-CPU
    float* a = (float*)calloc(m*n*sizeof(int), sizeof(float));
    float* b = (float*)calloc(n*k*sizeof(int), sizeof(float));
    float* c = (float*)calloc(m*k*sizeof(int), sizeof(float));
	float* c_linear = (float*)calloc(m*k*sizeof(int), sizeof(float));

    // Инициализация массивов
    for (int i = 0; i < m; i++) 
	{
        for (int j = 0; j<n; j++)
	    a[i*n+j] = rand()%10; 
    }
    for (int i = 0; i < n; i++) 
	{
        for (int j=0; j<k; j++)
	    b[i*k+j] = rand()%10; 
    }

    for (int i = 0; i < m; i++) 
	{
        for (int j=0; j<k; j++)
		{
	   		c[i*k+j]=0;
			c_linear[i*k+j]=0;
		} 
	}    
    printf("\n");
	
	// Вывод матриц
/*
    for (int i = 0; i < m; i++) 
	{
        for (int j=0; j<n; j++)
        {
             printf("%f", a[i*n+j]); 
             printf("  ");
        }   
	printf("\n");
    }
    printf("\n");
    for (int i = 0; i < n; i++) 
	{
        for (int j=0; j<k; j++)
        {
             printf("%f", b[i*k+j]); 
             printf("  ");
        }

	printf("\n");
    }
    printf("\n");
    for (int i = 0; i < m; i++) 
	{
        for (int j=0; j<k; j++)
        {
             printf("%f", c[i*k+j]); 
             printf("  ");
        }
	printf("\n");    
    }

*/
    // Выделение памяти на устройстве
    float* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, m*n*sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, n*k*sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, m*k*sizeof(float));
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
    cuerr = cudaMemcpy(adev, a, m*n*sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(bdev, b, n*k*sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
	cuerr = cudaMemcpy(cdev, c, m*k*sizeof(float), cudaMemcpyHostToDevice);
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
	//GRID_SIZE, BLOCK_SIZE
	dim3 DimGrid(m / 32 + 1, k / 32 + 1, 1);
    dim3 DimBlock(32, 32, 1);
    mulKernel <<< DimGrid, DimBlock >>> (cdev, adev, bdev, m, n, k);

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
    cuerr = cudaMemcpy(c, cdev, m*k*sizeof(float), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

/*
 	printf("\n");
    for (int i = 0; i < m; i++) 
	{
        for (int j=0; j<k; j++)
        {
             printf("%f", c[i*k+j]); 
             printf("  ");
        }
	printf("\n");    
    }
*/


    // Расчет времени
    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
	double gpu_time=gpuTime;
    printf("Time gpu: %.9f ms\n", gpu_time);
    
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    

//
//последовательный алгоритм
//

	clock_t start_linear = clock();
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < k; j++)
		{
			for (int q = 0; q < n; q++)
			{
				c_linear[i*k+j] += a[i*n+q] * b[q*k+j];
			}
		}
	}        
	clock_t end_linear = clock();
	double linear_time = ((double)(end_linear - start_linear) / CLOCKS_PER_SEC)*1000;
	printf("Time linear: %.9f ms\n", linear_time);

	float error = 0;
    for(int i = 0; i< m*k; i++) {
        c[i] -= c_linear[i];
        if(abs(c[i]) > error) error = abs(c[i]);
    }
	
	printf("Speedup: %.9f\n", linear_time/gpu_time);
	printf("Error: %.9f\n", error );

	free(a);
    free(b);
    free(c);
	free(c_linear);
    return 0;

}