#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cuda_runtime.h>

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
    {
        if (severity <= Severity::kVERBOSE)
        {
            printf("%d: %s\n", severity, msg);
        }
    }
};

nvinfer1::Weights make_weights(float *ptr, int n)
{
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

bool build_model_with_trt_api()
{
    // 使用trt C++ api构建network
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    // logger是必要的，用来捕捉warning和info等
    TRTLogger logger;
    // 需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    // 创建网络定义，其中createNetworkV2(1)表示采用显性batch size
    // 新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    // 构建一个模型
    /*
        Network definition:

        image
          |
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */
    // 输入是一个3通道的vector（可以理解为单像素图片）
    const int num_input = 3;
    // 输出两通道
    const int num_output = 2;
    // 前3个给w1的rgb，后3个给w2的rgb
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};

    // 使用trt C++ api定义网络并进行参数赋值
    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 2);
    //添加全连接层
    // 注意对input进行了解引用
    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
    // 注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    auto prob = network->addActivation(*(layer1->getOutput(0)), nvinfer1::ActivationType::kSIGMOID);

    // 将我们需要的prob标记为输出
    network->markOutput(*(prob->getOutput(0)));
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); // 256Mib
    config->setMaxWorkspaceSize(1 << 28);
    // 推理时 batchSize = 1
    builder->setMaxBatchSize(1);

    // ----------------------------- 3. 生成engine模型文件 -----------------------------
    // TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return -1;
    }

    // ----------------------------- 4. 序列化模型文件并存储 -----------------------------
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    return true;
}

bool build_model_with_trt_api_cnn_dynamic_input()
{
    TRTLogger logger;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    const int num_input = 1;
    const int num_output = 1;
    float layer1_weight_values[] = {
        1.0, 2.0, 3.1,
        0.1, 0.1, 0.1,
        0.2, 0.2, 0.2};
    float layer1_bias_values[] = {0.0};
    nvinfer1::ITensor *input = network->addInput("Image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 9);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 1);
    auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
    layer1->setPadding(nvinfer1::DimsHW(1, 1));
    auto prob = network->addActivation(*(layer1->getOutput(0)), nvinfer1::ActivationType::kRELU);
    network->markOutput(*(prob->getOutput(0)));

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    // 配置暂存存储器，用于layer实现的临时存储，也用于保存中间激活值
    config->setMaxWorkspaceSize(1 << 28);

    // --------------------------------- 2.1 关于profile ----------------------------------
    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    // 配置最小允许1 x 1 x 3 x 3
    // 配置最常尺寸1 x 1 x 3 x 3
    // 配置最大允许10 x 1 x 5 x 5
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }

    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("engine_cnn.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    return true;
}

std::vector<unsigned char> load_file(const std::string &file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char *)&data[0], length);
    }
    in.close();
    return data;
}

void inference()
{
    // ------------------------------ 1. 准备模型并加载   ----------------------------
    TRTLogger logger;
    auto engine_data = load_file("engine.trtmodel");
    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    // 创建推理时使用的context
    nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();

    // 创建CUDA流，以确定这个batch的推理是独立的
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
    float input_data_host[] = {1, 2, 3};
    float *input_data_device = nullptr;
    float output_data_host[2];
    float *output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 用一个指针数组指定input和output在gpu中的指针。
    float *bindings[] = {input_data_device, output_data_device};

    // ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ------------------------------ 4. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
}

void inference_cnn()
{
    TRTLogger logger;
    auto engine_data = load_file("engine_cnn.trtmodel");
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    float input_data_host[] = {
        // batch 0
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,

        // batch 1
        -1, 1, 1,
        1, 0, 1,
        1, 1, -1};
    float *input_data_device = nullptr;

    // 3x3输入，对应3x3输出
    int ib = 2;
    int iw = 3;
    int ih = 3;
    float output_data_host[ib * iw * ih];
    float *output_data_device = nullptr;
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
    float *bindings[] = {input_data_device, output_data_device};
    bool success = context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // ------------------------------- 4. 输出结果 -------------------------------
    for (int b = 0; b < ib; ++b)
    {
        printf("batch %d. output_data_host = \n", b);
        for (int i = 0; i < iw * ih; ++i)
        {
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if ((i + 1) % iw == 0)
                printf("\n");
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(output_data_device);
    cudaFree(input_data_device);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

int main()
{
    if (!build_model_with_trt_api())
    {
        return -1;
    }
    inference();
    printf("=================================\n");
    if (!build_model_with_trt_api_cnn_dynamic_input())
    {
        return -1;
    }
    inference_cnn();
    printf("Done.\n");
    return 0;
}