#include "EasyBMP.h"
#include <iostream>
#include <vector>
#include <algorithm> 
#include <string>
#include <iomanip>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
// G ������ ���������� ����� ���������
// arrayOutput �������� ������
// texObj ���� ��������
// _width - width image
// _ height - height image
// sigmaR - �������� sirmaR

__global__ void kernel(float* G, float* arrayOutput, cudaTextureObject_t texObj, int width, int height, float sigmaR) {

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	float a0 = tex2D<float>(texObj, index_x, index_y);
	float hai = 0;
	float ai;
	int index = 0;
	float k = 0;
	if ((index_x < width) && (index_y < height))
	{
		for (int i = index_x - 1; i <= index_x + 1; i++)
		{
			for (int j = index_y-1; j <= index_y + 1; j++)
			{
				ai = tex2D<float>(texObj, i, j);
				float rai = exp((pow(ai - a0, 2.0)) / (pow(sigmaR, 2.0)));
				hai += ai * G[index] * rai;
				k += G[index] * rai;
				++index;			
			}
		}
		arrayOutput[(index_x)  + (index_y)* (width)] = (hai / k);
	}
}

void moveCursor(std::ostream& os, int col, int row)
{
  os << "\033[" << col << ";" << row << "H";
}

void draw_frame() {
	// ����� ������ ������� ����� ������� �����, ������� 201, �.�. ����������� ����������� ����������� ASCII � ���� ������� ���
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

void GetGConst(float* G_CPU, float sigmaD) {
	int index = 0;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			G_CPU[index++] = exp((pow(i, 2) + pow(j, 2)) / (-1*pow(sigmaD,2)));
		}
	}
}

int BilateralFilter(vector<vector<int>> image, int verctical, int gorizontal, float sigmaR, float* G_CPU) {
	float a0 = image[verctical][gorizontal];
	float hai = 0;
	float ai;
	int index = 0;
	float k = 0;
	for (int i = verctical - 1; i <= verctical + 1; i++)
	{
		for (int j = gorizontal - 1; j <= gorizontal + 1; j++)
		{
			ai = image[i][j];
			float rai = exp((pow(ai - a0, 2)) / (pow(sigmaR, 2)));
			hai += ai * G_CPU[index] * rai;
			k += G_CPU[index] * rai;
			++index;
		}
	}
	return (int)(hai / k);
}

vector<vector<int>> transformationImage(vector<vector<int>> image, float sigmaR, float* G_CPU) {
	vector<vector<int>> output(image.size(), vector <int>(image[0].size()));
	int last_procent = -1;
	for (int i = 1; i < (image.size() - 1); i++)
	{
		int procent = (int)(((float)i / (float)(image.size() - 2)) * 100);
		if (last_procent != procent) {
			last_procent = procent;
			PrintProcent(procent);
		}
		for (int j = 1; j < (image[0].size() - 1); j++)
		{
			output[i][j] = BilateralFilter(image, i, j, sigmaR, G_CPU);
		}
	}
	return output;
}


bool IsCudaSuccess(cudaError_t cudaError, const char* message)
{
// �� ����� �������� �� cout, �� ���� ��� �� stderr, ��� ���� ������ (��������) -_-
	if (cudaError != cudaSuccess) {
		fprintf(stderr, message);
		fprintf(stderr, cudaGetErrorString(cudaError));
		fprintf(stderr, "\n");
		return false;
	}
	return true;
}

int main(int argc, char* argv[])
{
	system ("clear");
	draw_frame();
	float sigmaD = atof(argv[1]);
  	float sigmaR = atof(argv[2]);
	if (sigmaD < 0 || sigmaR < 0 )
	{
		cout << "Check the sigmaD and sigmaR parameters" << endl;
		return 0;
	}

	BMP Input;
	Input.ReadFromFile("input.bmp");
	int width = Input.TellWidth();
	int height = Input.TellHeight();

	vector<vector<int>> a(width + 2, vector <int>(height + 2));
	float* h_data = (float*)malloc(width * height * sizeof(float));
	float* G_CPU = (float*)malloc(9 * sizeof(float));
	float* G_GPU;

	GetGConst(G_CPU, sigmaD);

	cudaError_t cuerr = cudaMalloc(&G_GPU, 9 * sizeof(float));
	if (!IsCudaSuccess(cuerr, "Cannot allocate device Ouput array for G_GPU: ")) return 0;

	cuerr = cudaMemcpy(G_GPU, G_CPU, 9 * sizeof(float), cudaMemcpyHostToDevice);
	if (!IsCudaSuccess(cuerr, "Cannot copy a array from device to host: ")) return 0;
	
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			int temp = (int)floor(0.3 * Input(i, j)->Red + 0.59 * Input(i, j)->Green + 0.11 * Input(i, j)->Blue);
			a[i + 1][j + 1] = temp;
			h_data[i * height + j] = temp;
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


	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray_t arrayInput;
	float* arrayOutput;

	cuerr = cudaMalloc(&arrayOutput, width * height * sizeof(float));
	if (!IsCudaSuccess(cuerr, "Cannot allocate device Ouput array for a: ")) return 0;

	cuerr = cudaMallocArray(&arrayInput, &channelDesc, width, height);
	if (!IsCudaSuccess(cuerr, "Cannot allocate device Input array for a:")) return 0;

	cuerr = cudaMemcpy2DToArray(arrayInput, 0, 0, h_data, (width) * sizeof(float), (width) * sizeof(float), (height), cudaMemcpyHostToDevice);
	if (!IsCudaSuccess(cuerr, "Cannot copy a array2D from host to device: ")) return 0;

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = arrayInput;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp; // �� �������� �������� ����� ������������ ��������� ��������
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0; // �� ������������ ��������������� ���������

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cuerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	if (!IsCudaSuccess(cuerr, "Cannot create TextureObject: ")) return 0;

	// �������� ������������ �������
  	cudaEvent_t start, stop;
  	float gpuTime = 0.0f;
  	cuerr = cudaEventCreate(&start);
	if (!IsCudaSuccess(cuerr, "Cannot create CUDA start event: ")) return 0;

   	cuerr = cudaEventCreate(&stop);
	if (!IsCudaSuccess(cuerr, "Cannot create CUDA end event: ")) return 0;
   	
	dim3 BLOCK_SIZE(32, 32, 1);
	dim3 GRID_SIZE(height  / 32 + 1, width/ 32 + 1, 1);

	// ��������� ����� ������
    	cuerr = cudaEventRecord(start, 0);
    	if (cuerr != cudaSuccess) {
        	fprintf(stderr, "Cannot record CUDA event: %s\n",
            	cudaGetErrorString(cuerr));
        	return 0;
    	}

	kernel << <GRID_SIZE, BLOCK_SIZE >> > (G_GPU, arrayOutput, texObj, width, height, sigmaR);
	
	cuerr = cudaGetLastError();
	if (!IsCudaSuccess(cuerr, "Cannot launch CUDA kernel: ")) return 0;

	cuerr = cudaDeviceSynchronize();
	if (!IsCudaSuccess(cuerr, "Cannot synchronize CUDA kernel: ")) return 0;

    	cuerr = cudaEventRecord(stop, 0);
	if (!IsCudaSuccess(cuerr, "Cannot copy c array from device to host: ")) return 0;

	cuerr = cudaMemcpy(h_data, arrayOutput, width * sizeof(float) * height, cudaMemcpyDeviceToHost);
	if (!IsCudaSuccess(cuerr, "Cannot copy a array from device to host: ")) return 0;	

	struct timespec mt1, mt2; 
  	long double tt;
	clock_gettime(CLOCK_REALTIME, &mt1);

	a = transformationImage(a, sigmaR, G_CPU);
	
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
		{	
			ebmpBYTE color = (ebmpBYTE)h_data[i*height + j];
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