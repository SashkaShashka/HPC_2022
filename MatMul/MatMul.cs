using DotNumerics.LinearAlgebra;
using System.Diagnostics;


List<long> timeLA = new List<long>();
List<long> timeMyA = new List<long>();
File.WriteAllText("output.txt", "Size Matrix \n");
for (int i = 32; i <= 2048 ; i*=2)
{
    var B = Matrix.Random(i, i);
    var A = Matrix.Random(i, i);
    var _a = A.CopyToArray();
    var _b = B.CopyToArray();
    Console.WriteLine("Размер матрицы: {0}",i);
    Stopwatch stopwatch = new Stopwatch();
    stopwatch.Start();
    var C = A * B;
    stopwatch.Stop();
    timeLA.Add(stopwatch.ElapsedTicks);
    Console.WriteLine("LinearAlgebra: {0} ms. {1} tick", stopwatch.ElapsedMilliseconds, stopwatch.ElapsedTicks);


    stopwatch.Reset();
    stopwatch.Start();
    Multiplication(_a, _b);
    stopwatch.Stop();
    timeMyA.Add(stopwatch.ElapsedTicks);
    Console.WriteLine("linear Algorithm: {0} ms. {1} tick", stopwatch.ElapsedMilliseconds, stopwatch.ElapsedTicks);
    Console.WriteLine();
    File.AppendAllText("output.txt", i +" \n");
}
File.AppendAllText("output.txt", "LinearAlgebra Libraly \n");
foreach (var item in timeLA)
{
    File.AppendAllText("output.txt", item.ToString() + "\n");
}
File.AppendAllText("output.txt", "My Linear Algorithm \n");
foreach (var item in timeMyA)
{
    File.AppendAllText("output.txt", item.ToString() + "\n");
}





static double[,] Multiplication(double[,] a, double[,] b)
{
    if (a.GetLength(1) != b.GetLength(0)) throw new Exception("Матрицы нельзя перемножить");
    double[,] r = new double[a.GetLength(0), b.GetLength(1)];
    for (int i = 0; i < a.GetLength(0); i++)
    {
        for (int j = 0; j < b.GetLength(1); j++)
        {
            for (int k = 0; k < b.GetLength(0); k++)
            {
                r[i, j] += a[i, k] * b[k, j];
            }
        }
    }
    return r;
}
static void Print(double[,] a)
{
    for (int i = 0; i < a.GetLength(0); i++)
    {
        for (int j = 0; j < a.GetLength(1); j++)
        {
            Console.Write("{0} ", a[i, j]);
        }
        Console.WriteLine();
    }
}