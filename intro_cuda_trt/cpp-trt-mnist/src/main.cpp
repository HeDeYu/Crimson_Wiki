
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;


#define SIMLOG(type, ...)                        \
    do{                                          \
        printf("[%s:%d]%s: ", __FILE__, __LINE__, type); \
        printf(__VA_ARGS__);                     \
        printf("\n");                            \
    }while(0)

#define INFO(...)   SIMLOG("info", __VA_ARGS__)

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            SIMLOG(severity_string(severity), "%s", msg);
        }
    }
};

struct Matrix{
    vector<float> data;
    int rows = 0, cols = 0;

    void resize(int rows, int cols){
        this->rows = rows;
        this->cols = cols;
        this->data.resize(rows * cols * sizeof(float));
    }

    bool empty() const{return data.empty();}
    int size() const{ return rows * cols; }
    float* ptr() const{return (float*)this->data.data();}
};

struct Model{
    vector<Matrix> weights;
};

void print_image(const vector<unsigned char>& a, int rows, int cols, const char* format = "%3d"){
    INFO("Matrix[%p], %d x %d", &a, rows, cols);

    char fmt[20];
    sprintf(fmt, "%s,", format);

    for(int i = 0; i < rows; ++i){

        printf("row[%02d]: ", i);
        for(int j = 0; j < cols; ++j){
            int index = (rows - i - 1) * cols + j;
            printf(fmt, a.data()[index * 3 + 0]);
        }
        printf("\n");
    }
}

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

bool load_model(Model& model){
    model.weights.resize(4);

    const int weight_shapes[][2] = {
        {1024, 784},
        {1024, 1},
        {10, 1024},
        {10, 1}
    };

    for(int i = 0; i < model.weights.size(); ++i){
        char weight_name[100];
        sprintf(weight_name, "%d.weight", i);

        auto data = load_file(weight_name);
        if(data.empty()){
            INFO("Load %s failed.", weight_name);
            return false;
        }

        auto& w = model.weights[i];
        int rows = weight_shapes[i][0];
        int cols = weight_shapes[i][1];
        if(data.size() != rows * cols * sizeof(float)){
            INFO("Invalid weight file: %s", weight_name);
            return false;
        }

        w.resize(rows, cols);
        memcpy(w.ptr(), data.data(), data.size());
    }
    return true;
}

Matrix bmp_data_to_normalize_matrix(const vector<unsigned char>& data){

    Matrix output;
    const int std_w = 28;
    const int std_h = 28;
    if(data.size() != std_w * std_h * 3){
        INFO("Invalid bmp file, must be %d x %d @ rgb 3 channels image", std_w, std_h);
        return output;
    }
    output.resize(1, std_w * std_h);

    const unsigned char* begin_ptr = data.data();
    float* output_ptr = output.ptr();
    for(int i = 0; i < std_h; ++i){
        const unsigned char* image_row_ptr = begin_ptr + (std_h - i - 1) * std_w * 3;
        float* output_row_ptr = output_ptr + i * std_w;
        for(int j = 0; j < std_w; ++j){
            // normalize
            output_row_ptr[j] = (image_row_ptr[j * 3 + 0] / 255.0f - 0.1307f) / 0.3081f;;
        }
    }
    return output;
}

nvinfer1::Weights model_weights_to_trt_weights(const Matrix& model_weights){

    nvinfer1::Weights output;
    output.type = nvinfer1::DataType::kFLOAT;
    output.values = model_weights.ptr();
    output.count = model_weights.size();
    return output;
}

TRTLogger logger;
void do_trt_build_engine(const Model& model, const string& save_file){

    /*
        Network is:

        image
          |
        linear (fully connected)  input = 784, output = 1024, bias = True
          |
        relu
          |
        linear (fully connected)  input = 1024, output = 10, bias = True
          |
        sigmoid
          |
        prob
    */

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, 784, 1, 1));
    nvinfer1::Weights layer1_weight = model_weights_to_trt_weights(model.weights[0]);
    nvinfer1::Weights layer1_bias = model_weights_to_trt_weights(model.weights[1]);
    auto layer1 = network->addFullyConnected(*input, model.weights[0].rows, layer1_weight, layer1_bias);
    auto relu1 = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU);

    nvinfer1::Weights layer2_weight = model_weights_to_trt_weights(model.weights[2]);
    nvinfer1::Weights layer2_bias = model_weights_to_trt_weights(model.weights[3]);
    auto layer2 = network->addFullyConnected(*relu1->getOutput(0), model.weights[2].rows, layer2_weight, layer2_bias);
    auto prob = network->addActivation(*layer2->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    
    network->markOutput(*prob->getOutput(0));
    config->setMaxWorkspaceSize(1 << 28);
    builder->setMaxBatchSize(1);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        INFO("Build engine failed.");
        return;
    }

    nvinfer1::IHostMemory* model_data = engine->serialize();
    ofstream outf(save_file, ios::binary | ios::out);
    if(outf.is_open()){
        outf.write((const char*)model_data->data(), model_data->size());
        outf.close();
    }else{
        INFO("Open %s failed", save_file.c_str());
    }

    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}

void do_trt_inference(const string& model_file){

    auto engine_data = load_file(model_file);
    if(engine_data.empty()){
        INFO("engine_data is empty");
        return;
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        INFO("Deserialize cuda engine failed.");
        return;
    }

    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    const char* image_list[] = {"5.bmp", "6.bmp"};
    int num_image = sizeof(image_list) / sizeof(image_list[0]);
    const int num_classes = 10;
    for(int i = 0; i < num_image; ++i){

        const int bmp_file_head_size = 54;
        auto file_name  = image_list[i];
        auto image_data = load_file(file_name);
        if(image_data.empty() || image_data.size() != bmp_file_head_size + 28*28*3){
            INFO("Load image failed: %s", file_name);
            continue;
        }

        image_data.erase(image_data.begin(), image_data.begin() + bmp_file_head_size);
        auto image = bmp_data_to_normalize_matrix(image_data);
        float* image_device_ptr = nullptr;
        cudaMalloc(&image_device_ptr, image.size() * sizeof(float));
        cudaMemcpyAsync(image_device_ptr, image.ptr(), image.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

        float* output_device_ptr = nullptr;
        cudaMalloc(&output_device_ptr, num_classes * sizeof(float));

        float* bindings[] = {image_device_ptr, output_device_ptr};
        bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
        float predict_proba[num_classes];
        cudaMemcpyAsync(predict_proba, output_device_ptr, num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // release memory
        cudaFree(image_device_ptr);
        cudaFree(output_device_ptr);

        int predict_label  = std::max_element(predict_proba, predict_proba + num_classes) - predict_proba;
        float predict_prob = predict_proba[predict_label];

        print_image(image_data, 28, 28);
        INFO("image matrix: %d x %d", image.rows, image.cols);
        INFO("%s predict: %d, confidence: %f", file_name, predict_label, predict_prob);

        printf("Press 'Enter' to next, Press 'q' to quit: ");
        int c = getchar();
        if(c == 'q')
            break;
    }

    INFO("Clean memory");
    cudaStreamDestroy(stream);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
}   

int main(){

    Model model;
    if(!load_model(model))
        return 0;

    for(int i = 0; i < model.weights.size(); ++i){
        INFO("weight.%d shape = %d x %d", i, model.weights[i].rows, model.weights[i].cols);
    }

    auto trtmodel = "mnist.trtmodel";
    do_trt_build_engine(model, trtmodel);
    do_trt_inference(trtmodel);
    INFO("done.");
    return 0;
}