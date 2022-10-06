// ConsoleApplication1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
using namespace std;
#include <iostream>
#include <time.h> 

int main()
{
	for (int size_matrix = 32; size_matrix <= 2048; size_matrix*=2)
	{
        double** a, ** b, ** c;
        a = new double* [size_matrix];
        for (int i = 0; i < size_matrix; i++)
        {
            a[i] = new double[size_matrix];
            for (int j = 0; j < size_matrix; j++)
            {
                a[i][j]=(double)rand()/100;
            }
        }

        b = new double* [size_matrix];
        for (int i = 0; i < size_matrix; i++)
        {
            b[i] = new double[size_matrix];
            for (int j = 0; j < size_matrix; j++)
            {
                b[i][j] = (double)rand() / 100;
            }
        }
        cout << "Size matrix: " << size_matrix << endl;
        c = new double* [size_matrix];
        
        for (int i = 0; i < size_matrix; i++)
        {
            c[i] = new double[size_matrix];
            for (int j = 0; j < size_matrix; j++)
            {
                c[i][j] = 0;
            }
        }


        clock_t start = clock();
        for (int i = 0; i < size_matrix; i++)
        {
            c[i] = new double[size_matrix];
            for (int j = 0; j < size_matrix; j++)
            {
                for (int k = 0; k < size_matrix; k++)
                    c[i][j] += a[i][k] * b[k][j];;
            }
        }
        clock_t end = clock();
        cout.precision(20);
        double linear_time = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        printf("Time linear: %.9f ms\n", linear_time);
	}
}


