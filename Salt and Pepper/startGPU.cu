#include "EasyBMP.h"
#include <iostream>
#include <vector>
#include <algorithm> 
#include <string>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void kernel(float* arrayOutput, cudaTextureObject_t texObj, int width, int height) {

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int array[9];
	int k = 0;
	if ((index_x < width) && (index_y < height))
	{
		for (int i = index_x-1; i <= index_x + 1; i++)
		{
			for (int j = index_y-1; j <= index_y + 1; j++)
			{
				array[k] = (int)tex2D<float>(texObj, i, j);
				k++;
			}
		}
		for (int q = 0; q < 9; q++) 
		{
        		for (int w = 0; w < 8; w++) 
			{
            			if (array[w] > array[w + 1]) 
				{
               				float b = array[w]; // создали дополнительную переменную
                			array[w] = array[w+ 1]; // меняем местами
               				array[w + 1] = b; // значения элементов
            			}
        		}
    		}
		arrayOutput[(index_x)  + (index_y)* (width)] = array[4];

	}
}

void moveCursor(std::ostream& os, int col, int row)
{
  os << "\033[" << col << ";" << row << "H";
}

void draw_frame() {
	// любой символ который может сделать рамку, оставлю 201, т.к. корректного отображения расширенной ASCII в этой консоли нет
	cout << char(201);
	for (size_t i = 0; i < 100; i++)
		cout << char(201);	
	cout << char(201);
	cout << endl;
	cout << char(201);
	for (size_t i = 0; i < 49; i++)
		cout << "-";
	cout << "0%";
	for (size_t i = 0; i < 49; i++)
		cout << "-";
	cout << char(201);
	moveCursor(std::cout,3,1);
	cout << char(201);
	for (size_t i = 0; i < 100; i++)
	 	cout << char(201);
	cout << char(201);
	cout << endl;
}

void PrintProcent (int procent) {
	moveCursor(std::cout, 2, 2);
	for (size_t i = 0; i < procent; i++)
		cout << "|";	
	for (size_t i = 0; i < 100-(procent); i++)
		cout << "-";	
	cout << char(186);
	moveCursor(std::cout, 2,49);
	string procent_str = to_string(procent);
	if (procent < 49) cout << procent_str << "%";
	if (procent == 49) cout << procent_str[0] << procent_str[1] << "%";
	if (procent == 50) cout << procent_str << "%";
	if (procent > 50) cout << procent_str << "%";	cout << endl << endl;

}

int MedianFilter(vector<vector<int>> image, int verctical, int gorizontal) {
	vector<int> array;

	for (int i = verctical - 1; i <= verctical + 1; i++)
	{
		for (int j = gorizontal - 1; j <= gorizontal + 1; j++)
		{
			array.push_back(image[i][j]);
		}
	}
	sort(array.begin(), array.end());
	return array[4];
}

vector<vector<int>> transformationImage(vector<vector<int>> image) {
	vector<vector<int>> output(image.size(), vector <int>(image[0].size()));
	int last_procent = -1;

	//float count = 0;
	//float h = image.size()/100;
	for (int i = 1; i < image.size() - 1; i++)
	{
		int procent = (int)(((float)i / (float)(image.size() - 2)) * 100);
		if (last_procent != procent) {
			last_procent = procent;
			PrintProcent(procent);
		}
		//count +=h;
		//cout << setprecision(2) << count << endl;
		for (int j = 1; j < image[0].size() - 1; j++)
		{
			output[i][j] = MedianFilter(image, i, j);
		}
	}
	return output;
}

bool IsCudaSuccess(cudaError_t cudaError, const char* message)
{
	if (cudaError != cudaSuccess) {
		fprintf(stderr, message);
		fprintf(stderr, cudaGetErrorString(cudaError));
		fprintf(stderr, "\n");
		return false;
	}
	return true;
}
int main()
{
	system ("clear");
	draw_frame();
	BMP Input;
	Input.ReadFromFile("input.bmp");
	int width = Input.TellWidth();
	int height = Input.TellHeight();

	vector<vector<int>> a(width + 2, vector <int>(height + 2));

	// convert each pixel to grayscale using RGB->YUV
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			int Temp = (int)floor(0.299 * Input(i, j)->Red + 0.587 * Input(i, j)->Green + 0.114 * Input(i, j)->Blue);
			a[i + 1][j + 1] = Temp;
		}
	}

	for (size_t j = 1; j < height - 1; j++)
	{
		a[0][j] = a[1][j];
		a[width - 1][j] = a[width - 2][j];
	}
	for (size_t i = 1; i < width - 1; i++)
	{
		a[i][0] = a[i][1];
		a[i][height - 1] = a[i][height - 2];
	}
	a[0][0] = a[1][1];
	a[0][height - 1] = a[1][height - 2];
	a[width - 1][0] = a[width - 2][1];
	a[width - 1][height - 1] = a[width - 2][height - 2];

	float* h_data = (float*)malloc(width * height * sizeof(float));
	for (int i = 1; i < width+1; ++i)
		for (int j = 1; j < height+1; ++j)
			h_data[i * height + j] = a[i+1][j+1];

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray_t arrayInput;
	float* arrayOutput;

	cudaError_t cuerr = cudaMalloc((void**)&arrayOutput, width * height * sizeof(float));
	if (!IsCudaSuccess (cuerr, "Cannot allocate device Ouput array for a: ")) return 0;

	cuerr = cudaMallocArray(&arrayInput, &channelDesc, width, height);
	if (!IsCudaSuccess (cuerr, "Cannot allocate device Input array for a: ")) return 0;

	cuerr = cudaMemcpy2DToArray(arrayInput, 0, 0, h_data, (width) * sizeof(float), (width) * sizeof(float), (height), cudaMemcpyHostToDevice);
	if (!IsCudaSuccess (cuerr, "Cannot copy a array2D from host to device: ")) return 0;

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = arrayInput;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder; // режим Wrap
	texDesc.addressMode[1] = cudaAddressModeBorder;
	//texDesc.filterMode = cudaFilterModeLinear; // ближайшее значение
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0; // не использовать нормализованную адресацию

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cuerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	if (!IsCudaSuccess (cuerr, "Cannot create TextureObject: ")) return 0;

	// Создание обработчиков событий
  	cudaEvent_t start, stop;
  	float gpuTime = 0.0f;
  	cuerr = cudaEventCreate(&start);
	if (!IsCudaSuccess(cuerr, "Cannot create CUDA start event: ")) return 0;

   	cuerr = cudaEventCreate(&stop);
	if (!IsCudaSuccess(cuerr, "Cannot create CUDA end event: ")) return 0;

	dim3 BLOCK_SIZE(32, 32, 1);
	dim3 GRID_SIZE(height  / 32 + 1, width/ 32 + 1, 1);

	// Установка точки старта
    	cuerr = cudaEventRecord(start, 0);
    	if (cuerr != cudaSuccess) {
        	fprintf(stderr, "Cannot record CUDA event: %s\n",
            	cudaGetErrorString(cuerr));
        	return 0;
    	}

	kernel << <GRID_SIZE, BLOCK_SIZE >> > (arrayOutput, texObj, width, height);

	cuerr = cudaGetLastError();
	if (!IsCudaSuccess (cuerr, "Cannot launch CUDA kernel: ")) return 0;

	// Синхронизация устройств
	cuerr = cudaDeviceSynchronize();
	if (!IsCudaSuccess (cuerr, "Cannot synchronize CUDA kernel: ")) return 0;

	cuerr = cudaEventRecord(stop, 0);
	if (!IsCudaSuccess(cuerr, "Cannot copy c array from device to host: ")) return 0;

	cuerr = cudaMemcpy(h_data, arrayOutput, width * sizeof(float) * height, cudaMemcpyDeviceToHost);
	if (!IsCudaSuccess(cuerr, "Cannot copy a array from device to host: ")) return 0;	

	struct timespec mt1, mt2; 
  	long double tt;
	clock_gettime(CLOCK_REALTIME, &mt1);

	a = transformationImage(a);

	clock_gettime(CLOCK_REALTIME, &mt2);
  	tt=1000000000*(mt2.tv_sec - mt1.tv_sec)+(mt2.tv_nsec - mt1.tv_nsec);
  	cout << "Time CPU: " << tt/1000000000  << " second"<< endl;
	
  	cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
  	cout << "Time GPU: " << gpuTime /1000 << " second" << endl;
	cout << "SpeedUp: " << tt/(gpuTime*1000000) << endl;
	cout << "Width: " << width << endl;
	cout << "Height: "<< height << endl;

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{	//a[i + 1][j + 1]
			ebmpBYTE color = (ebmpBYTE)h_data[i * height + j];
			Input(i, j)->Red = color;
			Input(i, j)->Green = color;
			Input(i, j)->Blue = color;
		}
	}
	BMP Output;
	Output.ReadFromFile("input.bmp");
		
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			ebmpBYTE color = (ebmpBYTE)a[i + 1][j + 1];
			Output(i, j)->Red = color;
			Output(i, j)->Green = color;
			Output(i, j)->Blue = color;
		}
	}
	
	Input.WriteToFile("outputGPU.bmp");
	Output.WriteToFile("outputCPU.bmp");
	
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(arrayInput);
	cudaFree(arrayOutput);
	
	free(h_data);
	return 0;
}