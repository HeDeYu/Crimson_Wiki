#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory>   // for std::shared_ptr
#include <unistd.h> // for R_OK
#include <opencv2/opencv.hpp>

template <class T>
std::shared_ptr<T> make_nvshared(T *ptr)
{
    return std::shared_ptr<T>(ptr, [](T *p)
                              { p->destroy(); });
}

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

inline const char *severity_string(nvinfer1::ILogger::Severity t)
{
    switch (t)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:
        return "error";
    case nvinfer1::ILogger::Severity::kWARNING:
        return "warning";
    case nvinfer1::ILogger::Severity::kINFO:
        return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return "verbose";
    default:
        return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger // 需要实现ABC中的纯虚函数
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING)
            {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if (severity <= Severity::kERROR)
            {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else
            {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
};

bool exists(const std::string &path)
{

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

bool build_engine()
{
    std::string trt_file = "engine.trtmodel";
    if (exists(trt_file))
    {
        printf("Engine.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;
    // Note: builder is NOT ptr to IBuilder but std::shared_ptr<nvinfer1::IBuilder>

    // std::shared_ptr<nvinfer1::IBuilder> builder = make_nvshared<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    // auto builder = make_nvshared<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    // auto builder = make_nvshared<>(nvinfer1::createInferBuilder(logger));
    // auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto builder = make_nvshared<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto network = make_nvshared<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));

    auto parser = make_nvshared<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile("classifier.onnx", 1))
    {
        printf("parse onnx failed!\n");
        return false;
    }

    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    int input_channel = input_dims.d[1];
    int input_height = input_dims.d[2];
    int input_width = input_dims.d[3];
    int maxBatchSize = 16;

    // 描述输入动态尺寸的最小、最优、最大范围，并添加到配置中
    auto profile = builder->createOptimizationProfile();
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }

    auto model_data = make_nvshared<nvinfer1::IHostMemory>(engine->serialize());
    FILE *f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

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

std::vector<std::string> load_labels(const char* file){
    std::vector<std::string> lines;

    std:: ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open()){
        printf("open %d failed.\n", file);
        return lines;
    }
    
    std:: string line;
    while(getline(in, line)){
        lines.push_back(line);
    }
    in.close();
    return lines;
}

void inference()
{
    TRTLogger logger;
    auto model_data = load_file("engine.trtmodel");
    auto runtime = make_nvshared<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_data.data(), model_data.size()));
    auto execution_context = make_nvshared<>(engine->createExecutionContext());

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    int input_batch = 1;
    auto input_dims = execution_context->getBindingDimensions(0);
    int input_channel = input_dims.d[1];
    int input_height = input_dims.d[2];
    int input_width = input_dims.d[3];
    int input_numel = input_batch * input_channel * input_height * input_width;
    float *input_data_host = nullptr;
    float *input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    ///////////////////////////////////////////////////
    // image to float
    auto image = cv::imread("ng.jpg");
    float mean[] = {0.406, 0.456, 0.485};
    float std[] = {0.225, 0.224, 0.229};

    // 对应于pytorch的代码部分
    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.cols * image.rows;
    unsigned char *pimage = image.data;
    float *phost_b = input_data_host + image_area * 0;
    float *phost_g = input_data_host + image_area * 1;
    float *phost_r = input_data_host + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }
    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    const int num_classes = 1000;
    float output_data_host[num_classes];
    float *output_data_device = nullptr;
    checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

    input_dims.d[0] = input_batch;
    execution_context->setBindingDimensions(0, input_dims);
    float *bindings[] = {input_data_device, output_data_device};
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    float *prob = output_data_host;
    int predict_label = std::max_element(prob, prob + num_classes) - prob; // 确定预测类别的下标
    auto labels = load_labels("labels.imagenet.txt");
    auto predict_name = labels[predict_label];
    float confidence = prob[predict_label]; // 获得预测值的置信度
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
}

int main()
{
    if (!build_engine())
    {
        printf("Build engine failed!\n");
        return -1;
    }
    inference();
    printf("Done.\n");
    return 0;
}