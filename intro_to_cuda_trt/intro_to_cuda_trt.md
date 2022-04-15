精简版

## 1 `CUDA driver api`

头文件

```C++
#include <cuda.h>
```

`data type`

```C++
CUresult
CUdevice
CUcontext // 注意是指针类型，可初始化为nullptr
CUdeviceptr
```

`API`

```C++
cuInit()
cuDriverGetVersion()
cuDeviceGetName()
cuGetErrorName()
cuGetErrorString()
cuCtxCreate() // 创建context
cuCtxDestroy() // 销毁context
cuCtxGetCurrent() // 获取当前线程context栈的栈顶
cuCtxPushCurrent() // context入栈
cuCtxPopCurrent() // context出栈
cuDevicePrimaryCtxRetain() // 自动为一个设备关联一个context，并且这个设备只关联这个context
cuDevicePrimaryCtxRelease() // 上述context创建对应的释放
cuMemAlloc() // 在GPU上分配内存
cuMemAllocHost() // 在CPU上分配pinned内存，支持DMA
```



## 2 `CUDA runtime api`

### 2.1 相关头文件

```C++
#include <cuda_runtime.h>
```

数据类型、枚举与关键字

```C++
cudaError_t // 对驱动api中CUresult的封装
cudaStream_t
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaDeviceProp // 设备属性
cudaDeviceProp.maxGridSize
cudaDeviceProp.maxThreadsDim
cudaDeviceProp.warpSize
cudaDeviceProp.maxThreadsPerBlock
cudaDeviceProp.sharedMemPerBlock
dim3
__global__ // 注意C的stdio.h有cuda版本，但C++的iostream并没有
__shared__ // 静态共享内存，需要声明大小
extern __shared__ // 动态共享内存，不能声明大小

```

`api`

```C++
cudaGetDeviceCount() // 这个API不会触发context创建
cudaSetDevice() // 触发context创建

cudaMalloc() // 显存上创建并分配空间
cudaFree() // 释放显存
cudaMallocHost() // 内存上创建page-locker memory并分配空间
cudaFreeHost() // 释放page-locker memory

cudaStreamCreate() // 创建流对象
cudaStreamDestroy() // 销毁流对象
cudaStreamSynchronize() // 同步流对象

cudaMemcpy() // 数据复制
cudaMemcpyAsync() // 异步方式复制数据
    
cudaGetDeviceProperties() // 获取设备属性

__syncthreads() // 同一个block中的threads同步
```

示例

```C++
// CPU pageable memory创建分配
float *memory_host = new float[100];
// CPU pageable memory释放
delete[] memory_host;

// GPU显存创建分配
float *memory_device = nullptr;
cudaMalloc(&memory_device, 100 * sizeof(float));
// GPU显存释放
cudaFree(memory_device);

// CPU page-locked memory创建分配
float* memory_page_locked = nullptr;
cudaMallocHost(&memory_page_locked, 100 * sizeof(float));
cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost);
// CPU page-locked memory释放
cudaFreeHost(memory_page_locked);
```



TODO: cuda-runtime-api-1.5.2/1.6/1.8~1.14



## 3 使用`TRT c++ api`进行引擎构建与推理

### 3.1 相关头文件

```C++
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
```

数据类型

```C++
nvinfer1::ILogger // 创建builder的必要组件，常被用户继承
nvinfer1::IBuilder // 创建网络的组件
nvinfer1::IBuilderConfig // 告诉TRT如何优化网络结构实现加速
nvinfer1::ITensor
nvinfer1::Weights
nvinfer1::ICudaEngine // TRT推理引擎
nvinfer1::IHostMemory
nvinfer1::IRuntime // 推理时的builder角色
nvinfer1::IExecutionContext // 推理时的context
```

自定义`API`从数组数据构建`nvinfer1::Weights`数据

```
nvinfer1::Weights make_weights(float *ptr, int n)
{
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}
```

`API`

```C++
nvinfer1::ILogger logger; // logger
nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger); // logger创建builder
nvinfer1::IBuilderConfig *config = builder->createBuilderConfig(); // builder创建config
nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1); // builder创建network
nvinfer1::ITensor* input = network->addInput(...); // 添加input节点
nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
auto layer1 = network->addFullyConnected(...) // 添加全连接层节点
auto layer1 = network->addConvolution(...); // 添加卷积层节点
layer1->setPadding(nvinfer1::DimsHW(1, 1)); // 卷积节点padding操作设置
auto prob = network->addActivation(*(layer1->getOutput(0)), nvinfer1::ActivationType::kSIGMOID); // 添加激活层节点
network->markOutput(*(prob->getOutput(0))); // 设置网络的输出节点
config->setMaxWorkspaceSize(1 << 28); // ???

builder->setMaxBatchSize(1); // 设置推理时的batchsize=1

auto profile = builder->createOptimizationProfile(); // 如果模型有多个输入，则必须多个profile
profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3)); // 配置最小允许1 x 1 x 3 x 3
profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3)); // 配置最常尺寸1 x 1 x 3 x 3
profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5)); // 配置最大允许10 x 1 x 5 x 5
config->addOptimizationProfile(profile); // 把优化描述写进优化配置中

nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);//生成推理引擎
nvinfer1::IHostMemory *model_data = engine->serialize();//序列化推理引擎
fwrite(model_data->data(), 1, model_data->size(), f);//写入到文件f，用trtmodel扩展名

// 卸载顺序按照构建顺序倒序
model_data->destroy(); // 序列化文件
engine->destroy(); // 推理引擎
network->destroy(); // 网络
config->destroy(); // 配置
builder->destroy(); // 构建器

nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger); // 利用logger创建推理时runtime对象
nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()); // 把前述trtmodel文件加载后实例化推理引擎对象
nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();// 推理时需要的context
// 创建CUDA流，以确定这个batch的推理是独立的
cudaStream_t stream = nullptr;
cudaStreamCreate(&stream);


context->setBindingDimensions(0, nvinfer1::Dims(ib, 1, ih, iw));
// 创建绑定指针数组，由GPU输入输出指针构成
float *bindings[] = {input_data_device, output_data_device};
// 执行上下文，绑定指针数组，指定流入列
bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
// 结果数据搬回host
cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
// 同步流
cudaStreamSynchronize(stream);
// 倒序释放内存
cudaStreamDestroy(stream);
cudaFree(input_data_device);
cudaFree(output_data_device);
execution_context->destroy();
engine->destroy();
runtime->destroy();
```



## 4 使用`onnx`转`trt`进行引擎构建与推理

### 4.1 `pytorch`模型生成`onnx`文件

将模型设为推理状态，这一点很重要，因为像`dropout`, `batchnorm`这样的算子在推理和训练模式下的行为是不同的。

```python
torch.onnx.export(
    model, # torch.nn.Module具体类
    (dummy,), # 输入给model的参数，譬如dummy = torch.zeros(1, 1, 3, 3)，需要传递tuple
    "demo.onnx", # 储存的文件路径
    verbose=True, # 打印详细信息
    input_names=["image"], # 为输入和输出节点指定名称，方便后面查看或者操作
    output_names=["output"], 
    opset_version=11, # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
    dynamic_axes={
        "image": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }# 表示有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
     # 通常，我们只设置batch为动态，其他的避免动态
)
```

### 4.2 了解`onnx`文件内容

`model`下包含`graph`，`graph`下包含`input`，`output`，`node`，`initializer`。

`input`与`output`是`ValueInfoProto`类型，存储计算图的输入与输出。

`node`是`NodeProto`类型，存储计算图的`op`，例如`conv`，`bn`，`add`，`relu`等。

`initializer`是`TensorProto`类型，存储权重参数。

### 4.3 使用`onnx`的`python api`构建`onnx`文件

```python
import onnx.helper as helper
```

根据`onnx`文件内容，借助上述`python`模块提供的`api`创建`input`、`output`、`node`、`initializer`、`graph`等要素。

```python
# inputs与outputs列表，元素均是ValueInfoProto类型，对应使用make_value_info
inputs = [
    helper.make_value_info(
        name="image",
        type_proto=helper.make_tensor_type_proto(
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    )
]

outputs = [
    helper.make_value_info(
        name="output",
        type_proto=helper.make_tensor_type_proto(
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    )
]

# initializer存储权重系数，是TensorProto类型，对应使用make_tensor
initializer = [
    helper.make_tensor(
        name="conv.weight",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[1, 1, 3, 3],
        vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).tobytes(),
        raw=True
    ),
    helper.make_tensor(
        name="conv.bias",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[1],
        vals=np.array([0.0], dtype=np.float32).tobytes(),
        raw=True
    )
]

# node是NodeProto类型，对应使用make_node
# 注意节点的输入包括权重（initializer），输入对应input与initializer数据的name
# 自定义各个节点的output name，使各节点串连起来
nodes = [
    helper.make_node(
        name="Conv_0", # 节点名字
        op_type="Conv", # op类型
        # 各个输入的名字，结点的输入包含：输入和算子的权重。必有输入X和权重W，偏置B可以作为可选。
        inputs=["image", "conv.weight", "conv.bias"],  
        outputs=["3"],  
        # 其他字符串为节点的属性，attributes在官网被明确的给出了，标注了default的属性具备默认值。
        pads=[1, 1, 1, 1], 
        group=1,
        dilations=[1, 1],
        kernel_shape=[3, 3],
        strides=[1, 1]
    ),
    helper.make_node(
        name="ReLU_1",
        op_type="Relu",
        inputs=["3"],
        outputs=["output"]
    )
]

# 要素构建完毕，构建计算图，对应使用make_graph
graph = helper.make_graph(
    name="mymodel",
    inputs=inputs,
    outputs=outputs,
    nodes=nodes,
    initializer=initializer
)

# 如果名字不是ai.onnx，netron解析就不是太一样了
opset = [
    helper.make_operatorsetid("ai.onnx", version=11)
]

# producer主要是保持和pytorch一致
model = helper.make_model(graph, opset_imports=opset, producer_name="onnx", producer_version="xxx")
onnx.save_model(model, "my.onnx")
```



### 4.4 编辑`onnx`文件

读

```python
# 数据是以protobuf的格式存储的，因此当中的数值会以bytes的类型保存，通过np.frombuffer方法还原成类型为float32的ndarray
print(conv_weight.name, np.frombuffer(conv_weight.raw_data, dtype=np.float32))
```

增

```python
graph.initializer.append(xxx_tensor)
graph.node.insert(0, xxx_node)
```

删

```python
graph.node.remove(xxx_node)
```

改

```python
input_node.name = 'data'
```

千万注意删除节点的同时要考虑该节点输入输出指向的节点的重新连接，同时不要一边遍历一边移除节点，最好使用两遍遍历（顺序查找记录，倒序移除）。



### 4.5 使用`onnx`构建`trt`引擎进行推理

```c++
// 编译用的头文件
#include <NvInfer.h>
// 推理用的运行时头文件
#include <NvInferRuntime.h>
// onnx解析器的头文件
#include <NvOnnxParser.h>
// other .h files

// logger->builder->config/network
TRTLogger logger;
nvinfer1::IBuilder* builder=nvinfer1::createInferBuilder(logger);
nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
nvinfer1::INetworkDefinition* network=builder->createNetworkV2(1);
// 使用onnxparser导入onnx
nvonnxparser::IParser* parser=nvonnxparser::createParser(*network, logger);
parser->parseFromFile("demo.onnx", 1)
    
int maxBatchSize = 10;
auto profile = builder->createOptimizationProfile();
// 从onnx模型中提取出input尺寸信息
auto input_tensor = network->getInput(0);
int input_channel = input_tensor->getDimensions().d[1];
// 通过profile描述动态范围，并导入到config
profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
config->addOptimizationProfile(profile);
// 构建推理引擎
nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
// 保存
nvinfer1::IHostMemory *model_data = engine->serialize();
FILE *f = fopen("engine.trtmodel", "wb");
fwrite(model_data.->data(), 1, model_data->size(), f);
fclose(f)
// 释放指针
model_data->destroy();
parser->destroy();
engine->destroy();
network->destroy();
config->destroy();
builder->destroy();
```

### 4.6 使用`nvonnxparser`源码构建`trt`引擎进行推理

待完善

## 5 使用`pytorch`与`nvonnxparser`实现`trt`的`plugin`

### 5.1 完整流程

待完善

### 5.2 完整流程中的默认行为抽象以及`plugin`开发的简化

待完善

## 6 `Int8`量化

待完善

## 7 案例

```C++
// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}
```













