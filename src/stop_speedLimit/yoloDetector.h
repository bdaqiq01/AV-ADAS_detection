
#ifndef YOLO_DETECT
#define YOLO_DETECT 

#include <string>
#include <vector>
#include <opencv2/core.hpp> 
#include <NvInfer.h>
#include <cuda_runtime_api.h>



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

// generic yolov8/yolov11 trt detector (works for 1-class stop and multi-class speed limit)
class YoloDetector {

    public: 
    YoloDetector();
    YoloDetector(const std::vector<std::string>& classNames,
                 float confThreshold = 0.50f,
                 float nmsThreshold  = 0.45f);
    ~YoloDetector();

    bool loadEngine(const std::string& enginePath);
    std::vector<float> inferRaw(const cv::Mat& frame);
    std::vector<Detection> detect(const cv::Mat& frame);

    // optional setters in case caller wants to tweak after construction
    void setClassNames(const std::vector<std::string>& names) { classNames_ = names; }
    void setConfThreshold(float t) { confThreshold_ = t; }
    void setNmsThreshold(float t)  { nmsThreshold_  = t; }

    private:

    std::vector<char> readEngineFile(const std::string& enginePath);
    std::vector<float> preprocess(const cv::Mat& frame);

    Logger logger;

    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    std::string inputTensorName;
    std::string outputTensorName;

    int inputW;
    int inputH;
    int inputC;

    // derived from output tensor shape [1, 4 + numClasses, numPreds]
    int numClasses_;
    int numPreds_;
    
    size_t inputBytes;
    size_t outputBytes;

    void* dInput;
    void* dOutput;
    cudaStream_t stream;

    std::vector<std::string> classNames_;
    float confThreshold_;
    float nmsThreshold_;
};


#endif //YOLO_DETECT
