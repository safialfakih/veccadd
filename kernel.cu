
#include "common.h"
#include "timer.h"

__host__ __device__ float max(float a, float b) {
    if (a>b){
        return a;
    }else{
        return b;
    }
}

__global__ void vecMax_kernel(double* a, double* b, double* c, unsigned int M) {
    __global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i < N) {
                z[i] = max(x[i], y[i]);
                }
        }
}

void vecMax_gpu(double* a, double* b, double* c, unsigned int M) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**) &x_d, N*sizeof(float));
    cudaMalloc((void**) &y_d, N*sizeof(float));
    cudaMalloc((void**) &z_d, N*sizeof(float));

    // TODO
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // TODO



    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = N/numThreadsPerBlock;
    vecadd_kernel <<< numBlocks, numThreadsPerBlock >>> (x_d, y_d, z_d, N);

    // TODO




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // TODO


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    // TODO



    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

