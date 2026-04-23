
#ifndef STOP_DETECT
#define STOP_DETECT 

#include <string>
#include<vector>
#include<opencv2/core.hpp> 
#include <NvInfer.h>

struct Detection {
    cv::Rect box; 
    float confidence;
    int class_id;
    std::string label; 
};


class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

class stopSignDetect{

    public: 
    stopSignDetect();  //cons
    ~stopSignDetect(); //dest 

    bool loadEngine(const std::string& enginePath); //loads the model engine 
    std::vector<float> inferRaw(const cv::Mat& frame); //raw model output 
    std::vector<Detection> detect(const cv::Mat& frame); //


    private:

    std::vector<char> readEngineFile(const std::string& enginePath);

    Logger logger;

    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    std::string inputTensorName;
    std::string outputTensorName;

};


#endif //STOP_DETECT