#include <iostream>
#include <cuda.h>

//宏定义 #define <宏名>（<参数表>） <宏体>
#define checkDriver(op) __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char *op, const char *file, int line)
{
    if (code != CUresult::CUDA_SUCCESS)
    {
        const char *err_name = nullptr;
        const char *err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        std::cout << file << ":" << line << " failed." << std::endl;
        std::cout << "code=" << err_name << ", message=" << err_message << std::endl;
        return false;
    }
    return true;
}

int main(void)
{
    CUresult code = cuInit(0);

    /*
    测试获取当前cuda驱动的版本
    显卡、CUDA、CUDA Toolkit

        1. 显卡驱动版本，比如：Driver Version: 460.84
        2. CUDA驱动版本：比如：CUDA Version: 11.2
        3. CUDA Toolkit版本：比如自行下载时选择的10.2、11.2等；这与前两个不是一回事, CUDA Toolkit的每个版本都需要最低版本的CUDA驱动程序

        三者版本之间有依赖关系, 可参照https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        nvidia-smi显示的是显卡驱动版本和此驱动最高支持的CUDA驱动版本

     */
    std::cout << "===========================================" << std::endl;
    int driver_version = 0;
    // 返回值是CUresult类型
    checkDriver(cuDriverGetVersion(&driver_version));

    std::cout << "CUDA Driver version is " << driver_version << std::endl;

    CUdevice device = 0;
    char device_name[100];
    // 返回值是CUresult类型
    checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
    std::cout << "Device " << device << " name is " << device_name << std::endl;
    std::cout << "===========================================" << std::endl;
    // CUcontext 其实是 struct CUctx_st*（是一个指向结构体CUctx_st的指针）
    CUcontext ctxA = nullptr;
    CUcontext ctxB = nullptr;
    CUcontext current_ctx = nullptr;
    CUcontext popped_ctx = nullptr;

    // 告知要某一块设备上的某块地方创建 ctxA 管理数据。
    // 输入参数 参考 https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDA__CTX_g65dc0012348bc84810e2103a40d8e2cf.html
    checkDriver(cuCtxCreate(&ctxA, CU_CTX_SCHED_AUTO, device));
    checkDriver(cuCtxCreate(&ctxB, CU_CTX_SCHED_AUTO, device));
    std::cout << "ctxA=" << ctxA << std::endl;
    std::cout << "ctxB=" << ctxB << std::endl;
    checkDriver(cuCtxGetCurrent(&current_ctx));
    std::cout << "current context=" << current_ctx << std::endl;
    cuCtxPushCurrent(ctxA);
    cuCtxGetCurrent(&current_ctx);
    std::cout << "push A in stack, current context=" << current_ctx << std::endl;
    // pop A
    cuCtxPopCurrent(&popped_ctx);
    cuCtxGetCurrent(&current_ctx);
    std::cout << "pop, popped context=" << popped_ctx << std::endl;
    std::cout << "current context=" << current_ctx << std::endl;
    // pop B
    cuCtxPopCurrent(&popped_ctx);
    cuCtxGetCurrent(&current_ctx);
    std::cout << "pop, popped context=" << popped_ctx << std::endl;
    std::cout << "current context=" << current_ctx << std::endl;
    // pop A
    cuCtxPopCurrent(&popped_ctx);
    cuCtxGetCurrent(&current_ctx);
    std::cout << "pop, popped context=" << popped_ctx << std::endl;
    std::cout << "current context=" << current_ctx << std::endl;

    checkDriver(cuCtxDestroy(ctxA));
    checkDriver(cuCtxDestroy(ctxB));

    // 更推荐使用cuDevicePrimaryCtxRetain获取与设备关联的context
    // 注意这个重点，以后的runtime也是基于此, 自动为设备只关联一个context
    // 注意，这个context并不在上述的context栈内？？？？
    cuDevicePrimaryCtxRetain(&ctxA, device);
    std::cout << "ctxA=" << ctxA << std::endl;
    cuCtxGetCurrent(&current_ctx);
    std::cout << "current context=" << current_ctx << std::endl;
    cuDevicePrimaryCtxRelease(device);
    std::cout << "===========================================" << std::endl;
    CUcontext ctx = nullptr;
    cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, device);
    // 输入device prt向设备要一个100 byte的线性内存，并返回地址
    // 注意这是指向device的pointer，不能使用nullptr作为空指针初始化，只能用0
    CUdeviceptr device_memory_pointer = 0;
    cuMemAlloc(&device_memory_pointer, 100);
    std::cout << "device mem ptr=" << &device_memory_pointer << std::endl;
    std::cout << "===========================================" << std::endl;
    // 输入二级指针向host要一个100 byte的锁页内存，专供设备访问。
    // 参考 2.cuMemAllocHost.jpg 讲解视频：https://v.douyin.com/NrYL5KB/
    // 注意cuMemAllocHost分配出来的是pinned memory / page-locked memory
    float *host_page_locked_memory = nullptr;
    checkDriver(cuMemAllocHost((void **)&host_page_locked_memory, 100));
    std::cout << "host_page_locked_memory=" << host_page_locked_memory << std::endl;

    // 向page-locked memory 里放数据（仍在CPU上），可以让GPU可快速读取
    host_page_locked_memory[0] = 123;
    std::cout << "host_page_locked_memory[0]=" << host_page_locked_memory[0] << std::endl;

    /*
        记住这一点
        host page locked memory 声明的时候为float*型，可以直接转换为device ptr，这才可以送给cuda核函数（利用DMA(Direct Memory Access)技术）
        初始化内存的值: cuMemsetD32 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N )
        初始化值必须是无符号整型，因此需要将new_value进行数据转换：
        但不能直接写为:(int)value，必须写为*(int*)&new_value, 我们来分解一下这条语句的作用：
        1. &new_value获取float new_value的地址
        (int*)将地址从float * 转换为int*以避免64位架构上的精度损失
        *(int*)取消引用地址，最后获取引用的int值
     */
    float new_value = 555;
    cuMemsetD32((CUdeviceptr)host_page_locked_memory, *(int *)&new_value, 1);
    std::cout << "host_page_locked_memory[0]=" << host_page_locked_memory[0] << std::endl;

    checkDriver(cuCtxDestroy(ctx));
    std::cout << "Done..." << std::endl;

    return 0;
}