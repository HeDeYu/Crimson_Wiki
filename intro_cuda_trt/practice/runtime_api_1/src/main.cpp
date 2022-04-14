#include <iostream>
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

int main(void)
{
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    std::cout << "current ctx = " << ctx << std::endl;

    // cuda runtime是以cuda为基准开发的运行时库
    // cuda runtime所使用的CUcontext是基于cuDevicePrimaryCtxRetain函数获取的
    // 即，cuDevicePrimaryCtxRetain会为每个设备关联一个context，通过cuDevicePrimaryCtxRetain函数可以获取到
    // 而context初始化的时机是懒加载模式，即当你调用一个runtime api时，会触发创建动作
    // 也因此，避免了cu驱动级别的init和destroy操作。使得api的调用更加容易
    int device_cnt = 0;
    checkRuntime(cudaGetDeviceCount(&device_cnt));
    std::cout << "device count = " << device_cnt << std::endl;

    // 取而代之，是使用setdevice来控制当前上下文，当你要使用不同设备时
    // 使用不同的device id
    // 注意，context是线程内作用的，其他线程不相关的, 一个线程一个context stack
    int device_id = 0;
    std::cout << "set current device to " << device_id << std::endl;
    std::cout << "this api depends on CUcontext, create and set context." << std::endl;
    cudaSetDevice(device_id);
    // 注意，是由于set device函数是“第一个执行的需要context的函数”，
    // 它会执行cuDevicePrimaryCtxRetain
    // 并设置当前context，这一切都是默认执行的。
    // 注意：cudaGetDeviceCount是一个不需要context的函数
    // 你可以认为绝大部分runtime api都是需要context的
    // 第一个执行的cuda runtime函数，会创建context并设置上下文
    cuCtxGetCurrent(&ctx);
    std::cout << "current ctx = " << ctx << std::endl;
    std::cout << "===========================================" << std::endl;
    // CPU pageable memory创建分配
    float *memory_host = new float[100];
    memory_host[2] = 520.25;
    // GPU显存创建分配
    float *memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));
    // 从内存复制到显存
    cudaMemcpy(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice);

    // CPU page-locked memory创建分配
    float *memory_page_locked = nullptr;
    cudaMallocHost(&memory_page_locked, 100 * sizeof(float));
    cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost);
    std::cout << memory_page_locked[2] << std::endl;

    // CPU page-locked memory释放
    cudaFreeHost(memory_page_locked);
    // CPU pageable memory释放
    delete[] memory_host;
    // GPU显存释放
    checkRuntime(cudaFree(memory_device));
    std::cout << "===========================================" << std::endl;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    float *memory_device_2 = nullptr;
    checkRuntime(cudaMalloc(&memory_device_2, 100 * sizeof(float)));

    float *memory_host_2 = new float[100];
    memory_host_2[2] = 520.25;
    checkRuntime(cudaMemcpyAsync(memory_device_2, memory_host_2, sizeof(float) * 100, cudaMemcpyHostToDevice, stream));

    float *memory_page_locked_2 = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked_2, 100 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(memory_page_locked_2, memory_device_2, 100 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    std::cout << "before syn, data = " << memory_page_locked_2[2] << std::endl;
    checkRuntime(cudaStreamSynchronize(stream));
    std::cout << "after syn, data = " << memory_page_locked_2[2] << std::endl;
    
    delete[] memory_host_2;
    cudaFree(memory_device_2);
    cudaFreeHost(memory_page_locked_2);

    return 0;
}