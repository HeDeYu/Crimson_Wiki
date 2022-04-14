#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>


__global__ void test_print_kernel(const float *pdata, int ndata)
{
    /*    dims                 indexs
        gridDim.z            blockIdx.z
        gridDim.y            blockIdx.y
        gridDim.x            blockIdx.x
        blockDim.z           threadIdx.z
        blockDim.y           threadIdx.y
        blockDim.x           threadIdx.x

        Pseudo code:
        position = 0
        for i in 6:
            position *= dims[i]
            position += indexs[i]
    */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // std::cout << idx << " elem=" << pdata[idx] << " threadIdx.x=" << threadIdx.x << " blockIdx.x=" << blockIdx.x << " blockDim.x=" << blockDim.x << std::endl;
    printf("Element[%d] = %f, threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d, gridDim.x=%d\n", idx, pdata[idx], threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
    return;
}

__global__ void demo_kernel()
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("Run kernel. blockIdx = %d,%d,%d  threadIdx = %d,%d,%d\n",
               blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z);
    return;
}

__global__ void demo1_kernel()
{
    // 静态共享变量和动态共享变量在kernel函数内/外定义都行，没有限制
    printf("in demo1\n");
    const size_t static_shared_memory_num_element = 6 * 1024; // 6KB
    __shared__ char static_shared_memory[static_shared_memory_num_element];
    __shared__ char static_shared_memory2[2];
    extern __shared__ char dynamic_shared_memory[];
    extern __shared__ char dynamic_shared_memory2[];
    // 静态共享变量，定义几个地址随之叠加
    printf("static_shared_memory = %p\n", static_shared_memory);
    printf("static_shared_memory2 = %p\n", static_shared_memory2);
    // 动态共享变量，无论定义多少个，地址都一样
    printf("dynamic_shared_memory = %p\n", dynamic_shared_memory);
    printf("dynamic_shared_memory2 = %p\n", dynamic_shared_memory2);
    // __syncthreads();
    return;
}


__global__ void demo2_kernel(){
    __shared__ int shared_value1;
    __shared__ int shared_value2;
    if(threadIdx.x==0){
        if(blockIdx.x==0){
        shared_value1 = 123;
        shared_value2 = 55;
        }   
        else{
            shared_value1 = 331;
            shared_value2 = 8;
        }
    }
    // 等待block内的所有线程执行到这一步
    __syncthreads();
    printf("%d.%d. shared_value1 = %d[%p], shared_value2 = %d[%p]\n", 
        blockIdx.x, threadIdx.x,
        shared_value1, &shared_value1, 
        shared_value2, &shared_value2
    );
}



void test_print(const float *pdata, int ndata)
{
    std::cout << "in test_print():" << ndata << std::endl;k
    test_print_kernel<<<dim3(2, 1, 1), dim3(ndata / 2, 1, 1), 0, nullptr>>>(pdata, ndata);
    return;
}

void launch(int *grids, int *blocks)
{
    std::cout << "in launch():" << std::endl;
    dim3 grid_dims(grids[0], grids[1], grids[2]);
    dim3 block_dims(blocks[0], blocks[1], blocks[2]);
    demo_kernel<<<grid_dims, block_dims, 0, nullptr>>>();
    return;
}

void launch_()
{
    std::cout << "in launch_():" << std::endl;
    demo1_kernel<<<1, 1, 12, nullptr>>>();
    cudaDeviceSynchronize();
    demo2_kernel<<<2, 5, 0, nullptr>>>();
    cudaDeviceSynchronize();
    return;
}