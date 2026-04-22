class StopSignDetector {
public:
    bool loadEngine(const std::string& enginePath);
    std::vector<Detection> infer(const cv::Mat& frame);

private:
    // TensorRT objects
    // runtime, engine, context, cuda stream
    // input/output buffers
};
