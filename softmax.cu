/**********************************************************************************************************************************************************************
* softmax: softmax(xi) = exp(xi) / (exp(x0) + exp(x1) + exp(x2) + ........ + exp(xn))
* 
* Implementation contains 2 CUDA kernels
* 1. softmaxDenomKernel: calculates the denominator part of the equation using parallel reduction method.
* 2. softmaxKernel: calculates the final (numerator / denominator)
* 
* With 64 classes and total 10000384 data points below are the results
* 
*  Function Name    | Duration(us)| Compute Throughput(%) | Memory Throughput(%) | Registers(register/thread) | Grid Size      | Block Size (block)  | 
* ------------------|-------------|-----------------------|----------------------|----------------------------|----------------|---------------------|
* softmaxDenomKernel|   596.03    |       27.23           |         84.50        |             16             |   156256, 1, 1 |  64, 1, 1           |
* softmaxKernel     |   347.65    |       65.95           |         65.95        |             16             |   156256, 1, 1 |  64, 1, 1           |
* ---------------------------------------------------------------------------------------------------------------------------------------------------|
* 
* 
**********************************************************************************************************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <ctime>
#include <limits>

#define CHECK_CORRECTNESS   //Enable to check output correctness with CPU implementation

//declaration of wrapper function for cuda kernel
cudaError_t softmaxWithCuda();

#ifdef CHECK_CORRECTNESS
void softmaxCPU(float* out, float* in, float* denoms, int c, int b)
{
    for (int x1 = 0; x1 < b; x1++)
    {
        for (int x2 = 0; x2 < c; x2++)
        {
            out[(x1 * c) + x2] = exp(in[(x1 * c) + x2]) / denoms[x1];
        }
    }
}

void softmaxDenomCPU(float* out, const float* in, int c, int b)
{
    for (int x1 = 0; x1 < b; x1++)
    {
        float denom = 0.0f;
        for (int x2 = 0; x2 < c; x2++)
        {
            denom += exp(in[x1 * c + x2]);
        }
        out[x1] = denom;
    }
}
#endif


/*******************************************************************************
* softmaxDenomKernel calcuates the exponential sum of all the class values in the
* tensor or sum of all class values within a given batch
* It uses parallel reduction, each thread loads 2 values from the shared memory
* and adds them and picks the next 2 values after reduction of size by 2.
* Keep repeating the above until you reach 1 value.
*******************************************************************************/
__global__ void softmaxDenomKernel(float* out, const float* in)
{   
    //shared memory array to store accumulated values from each thread
    extern __shared__ float dData[];

    unsigned int threadId = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int blockId = blockIdx.x;
    unsigned int it = blockId * blockSize + threadId;

    //load 1 value per thread
    dData[threadId] = exp(in[it]);
    __syncthreads();

    /*run the loop till middle of the block since we will picking 
    2 values from lef and right to accumulate*/
    for (int x1 = blockSize / 2; x1 > 0; x1 >>= 1)
    {
        //to make sure thread is loading the correct value from left side
        if (threadId < x1)
        {
            /*pick a value from left half and add it to the corresponding value
            on the right half and store it in left half which will be picked 
             up in the next iteration*/
            dData[threadId] += dData[threadId + x1];
        }
        __syncthreads();
    }

    if (threadId == 0)
    {
        /*in the end the 1st value in the shared memory array iwll be the
        accumulated sum for the current batch*/
        out[blockId] = dData[threadId];
    }
}

/*******************************************************************************
* softmaxKernel calcuates the final output. Each thread will be reading 1 input 
* calculating the exponent of that and dividing it with the respective batch 
* accumulated denominator
*******************************************************************************/
__global__ void softmaxKernel(float *out, const float *in, const float *denoms)
{
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;

    out[(blockId * blockSize) + threadId] = exp(in[(blockId * blockSize) + threadId]) / denoms[blockId];
}

int main()
{
    // Add vectors in parallel.
    cudaError_t cudaStatus = softmaxWithCuda();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "softmaxWithCuda failed!");
        return 1;
    }

    /*cudaDeviceReset must be called before exiting in order for profiling and
    tracing tools such as Nsight and Visual Profiler to show complete traces.*/
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t softmaxWithCuda()
{
    std::srand(std::time(0));

    //pointers for device memories
    float *d_in = 0;
    float* d_denom = 0;
    float *d_out = 0;

    cudaError_t cudaStatus;

    //total num of data points
    int n = 10000384;
    //number of classes
    int c = 64;
    //number of blocks/batches
    int b = n / c;

    //allocating memories for host memory
    float *h_in = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    float* h_denom = (float*)malloc(b * sizeof(float));

#ifdef CHECK_CORRECTNESS
    /*allocating memeories on host memory for storing
    CPU reference output*/
    float *ref_out = (float*)malloc(n * sizeof(float));
    float* ref_denom = (float*)malloc(b * sizeof(float));
#endif

    float minVal = -2.0;
    float maxVal = 2.0f;
    float range = maxVal - minVal;

    //Generating random input values
    for (int x1 = 0; x1 < n; x1++)
    {
        float random_float_0_1 = (float)(std::rand()) / (float)RAND_MAX;
        h_in[x1] = minVal + (random_float_0_1 * range);
    }


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (input, intermediate denominators, output)
    cudaStatus = cudaMalloc((void**)&d_in, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_out, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_denom, b * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

#ifdef CHECK_CORRECTNESS
    softmaxDenomCPU(ref_denom, h_in, c, b);
#endif

    //number of classes as threads and batches as blocks
    int num_threads = c;
    int num_blocks = (n + num_threads - 1) / num_threads;
    // Launch a kernel on the GPU with one thread for each element.
    softmaxDenomKernel <<<num_blocks, num_threads, num_threads * sizeof(float) >>> (d_denom, d_in);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "softmaxDenomKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching softmaxDenomKernel!\n", cudaStatus);
        goto Error;
    }

#ifdef CHECK_CORRECTNESS
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_denom, d_denom, b * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "softmaxDenomkernel cudaMemcpy failed!");
        goto Error;
    }

    //Checking correctness of denominators with CPU reference
    int flag = 1;
    for (int x1 = 0; x1 < b; x1++)
    {
        float diff = abs(ref_denom[x1] - h_denom[x1]);
        if (diff > 0.009)
        {
            printf("Failure: softmaxDenomKernel output not matching at index %d, values are: (%f, %f)\n", x1, ref_denom[x1], h_denom[x1]);
            flag = -1;
            break;
        }
    }
    if (flag)
    {
        printf("Success: softmaxDenomKernel outputs are matching with cpu outputs.....!!!!!!!!!!\n");
    }
    else
    {
        goto Error;
    }

    //Final softmax on CPU
    softmaxCPU(ref_out, h_in, h_denom, c, b);
    /*Checking correctness of CPU output by adding the probablities
    and checking if they add upto 1*/
    for (int x1 = 0; x1 < b; x1++)
    {
        float sum = 0.0;
        for (int x2 = 0; x2 < c; x2++)
        {
            sum += ref_out[(x1 * c) + x2];
        }
        float diff = abs(sum - 1.0f);
        if (diff >  0.00009f)
        {
            printf("softmaxCPU invalid output, check CPU implementation, value = %f\n", diff);
            break;
        }
    }
#endif

    //number of classes as threads and batches as blocks
    num_threads = c;
    num_blocks = (n + num_threads - 1) / num_threads;
    // Launch a kernel on the GPU with one thread for each element.
    softmaxKernel<<<num_blocks, num_threads>>>(d_out, d_in, d_denom);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "softmaxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching softmaxKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

#ifdef CHECK_CORRECTNESS
    //Chekcing correctness of final output with CPU reference
    flag = 1;
    for (int x1 = 0; x1 < n; x1++)
    {
        float diff = abs(ref_out[x1] - h_out[x1]);
        if (diff > 0.0009)
        {
            printf("Failure: softmaxKernel output not matching at index %d, values are: (%f, %f)\n", x1, ref_out[x1], h_out[x1]);
            flag = -1;
            break;
        }
    }

    if (flag)
    {
        printf("Success: softmaxKernel outputs are matching with cpu outputs.....!!!!!!!!!!\n");
    }
#endif
    

Error:
    //freeing all the host and device allocated memories
    free(h_in);
    free(h_out);
    free(h_denom);
#ifdef CHECK_CORRECTNESS
    free(ref_denom);
    free(ref_out);
#endif
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_denom);
    
    return cudaStatus;
}
