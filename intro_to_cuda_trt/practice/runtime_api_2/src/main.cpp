#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

//宏定义 #define <宏名>（<参数表>） <宏体>
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        std::cout << file << ":" << line << " failed." << std::endl;
        std::cout << "code=" << err_name << ", message=" << err_message << std::endl;
        return false;
    }
    return true;
}

void test_print(const float *pdata, int ndata);

void launch(int* grids, int* blocks);

void launch_();

int main(void)
{
    // 创建并分配pageable memory与GPU memory
    int narray = 20;
    int array_bytes = sizeof(float) * narray;

    float *parray_host = new float[narray];
    float *parray_device = nullptr;
    cudaMalloc(&parray_device, array_bytes);

    // 给host memory赋值
    for (int i = 0; i < narray; ++i)
    {
        parray_host[i] = i;
    }
    // 复制数据到GPU
    cudaMemcpy(parray_device, parray_host, array_bytes, cudaMemcpyHostToDevice);
    test_print(parray_device, narray);
    // 等待GPU执行完毕
    cudaDeviceSynchronize();

    // 释放显存与内存
    cudaFree(parray_device);
    delete[] parray_host;
    std::cout << "===========================================" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // 通过查询maxGridSize和maxThreadsDim参数，得知能够设计的gridDims、blockDims的最大值
    // warpSize则是线程束的线程数量
    // maxThreadsPerBlock则是一个block中能够容纳的最大线程数，也就是说blockDims[0] * blockDims[1] * blockDims[2] <= maxThreadsPerBlock
    printf("prop.maxGridSize = %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("prop.maxThreadsDim = %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("prop.warpSize = %d\n", prop.warpSize);
    printf("prop.maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);

    int grids[] = {1, 1, 4};
    int blocks[] = {512, 2, 1};
    launch(grids, blocks);
    checkRuntime(cudaPeekAtLastError());   // 获取错误 code 但不清楚error
    checkRuntime(cudaDeviceSynchronize()); // 进行同步，这句话以上的代码全部可以异步操作
    std::cout << "===========================================" << std::endl;
    
    printf("prop.sharedMemPerBlock = %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    launch_();
    return 0;
}