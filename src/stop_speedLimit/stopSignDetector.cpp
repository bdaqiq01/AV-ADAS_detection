
#include "stopSignDetector.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/imgproc.hpp>

using namespace std; 

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING) {
        cout << msg << endl;
    }
}


stopSignDetect::stopSignDetect()
    : runtime(nullptr),
      engine(nullptr),
      context(nullptr),
      inputW(0),
      inputH(0),
      inputC(0),
      inputBytes(0),
      outputBytes(0),
      dInput(nullptr),
      dOutput(nullptr),
      stream(nullptr)
{
    cout << "stopSignDetect created" << endl;
}


stopSignDetect::~stopSignDetect()
{
    if (dInput) {
        cudaFree(dInput);
        dInput = nullptr;
    }

    if (dOutput) {
        cudaFree(dOutput);
        dOutput = nullptr;
    }

    if (stream) {
        cudaStreamDestroy(stream);
    }

    if (context) {
        delete context;
        context = nullptr;
    }

    if (engine) {
        delete engine;
        engine = nullptr;
    }

    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }

    cout << "stopSignDetect destroyed" << endl;
}





vector<char> stopSignDetect::readEngineFile(const string& enginePath)
{
    ifstream file(enginePath, ios::binary);

    if (!file) {
        cerr << "Failed to open engine file: " << enginePath << endl;
        return {};
    }

    file.seekg(0, ios::end);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);

    if (fileSize <= 0) {
        cerr << "Engine file is empty or invalid: " << enginePath << endl;
        return {};
    }

    vector<char> engineData(static_cast<size_t>(fileSize));
    file.read(engineData.data(), fileSize);

    if (!file) {
        cerr << "Failed to read full engine file: " << enginePath << endl;
        return {};
    }

    cout << "Read engine file: " << enginePath
         << " (" << fileSize << " bytes)" << endl;

    return engineData;
}

bool stopSignDetect::loadEngine(const string& enginePath)
{
    cout << "Loading engine..." << endl;

    vector<char> engineData = readEngineFile(enginePath);
    if (engineData.empty()) {
        cerr << "Engine file read failed." << endl;
        return false;
    }

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        cerr << "Failed to create TensorRT runtime." << endl;
        return false;
    }
    cout << "TensorRT runtime created." << endl;

    engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        cerr << "Failed to deserialize TensorRT engine." << endl;
        return false;
    }
    cout << "TensorRT engine deserialized." << endl;

    context = engine->createExecutionContext();
    if (!context) {
        cerr << "Failed to create execution context." << endl;
        return false;
    }
    cout << "Execution context created." << endl;

    int numIOTensors = engine->getNbIOTensors();
    cout << "Number of IO tensors: " << numIOTensors << endl;

    nvinfer1::Dims inputDims{};
    nvinfer1::Dims outputDims{};

    for (int i = 0; i < numIOTensors; i++) {
        const char* tensorName = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(tensorName);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(tensorName);

        cout << "Tensor " << i << ": " << tensorName << " | ";
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            cout << "INPUT";
            inputTensorName = tensorName;
            inputDims = dims;
        } else {
            cout << "OUTPUT";
            outputTensorName = tensorName;
            outputDims = dims;
        }

        cout << " | Shape: [";
        for (int d = 0; d < dims.nbDims; d++) {
            cout << dims.d[d];
            if (d < dims.nbDims - 1) cout << ", ";
        }
        cout << "]" << endl;
    }

    if (inputTensorName.empty() || outputTensorName.empty()) {
        cerr << "Could not identify input/output tensors." << endl;
        return false;
    }

    inputC = inputDims.d[1];
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];

    inputBytes = static_cast<size_t>(inputC) * inputH * inputW * sizeof(float);

    size_t outputCount = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        outputCount *= static_cast<size_t>(outputDims.d[i]);
    }
    outputBytes = outputCount * sizeof(float);

    cout << "Input tensor name: " << inputTensorName << endl;
    cout << "Output tensor name: " << outputTensorName << endl;
    cout << "Input bytes: " << inputBytes << endl;
    cout << "Output bytes: " << outputBytes << endl;

    cudaError_t err;

    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << endl;
        return false;
    }

    err = cudaMalloc(&dInput, inputBytes);
    if (err != cudaSuccess) {
        cerr << "Failed to allocate input buffer: " << cudaGetErrorString(err) << endl;
        return false;
    }

    err = cudaMalloc(&dOutput, outputBytes);
    if (err != cudaSuccess) {
        cerr << "Failed to allocate output buffer: " << cudaGetErrorString(err) << endl;
        return false;
    }

    cout << "CUDA buffers allocated successfully." << endl;

    return true;
}



vector<float> stopSignDetect::inferRaw(const cv::Mat& frame)
{
    cout << "inferRaw called frame size "
         << frame.cols << "x" << frame.rows << endl;

    vector<float> inputData = preprocess(frame);
    vector<float> outputData(outputBytes / sizeof(float));

    cudaError_t err;

    err = cudaMemcpyAsync(
        dInput,
        inputData.data(),
        inputBytes,
        cudaMemcpyHostToDevice,
        stream
    );
    if (err != cudaSuccess) {
        cerr << "cudaMemcpyAsync input failed: " << cudaGetErrorString(err) << endl;
        return {};
    }

    if (!context->setTensorAddress(inputTensorName.c_str(), dInput)) {
        cerr << "Failed to set input tensor address." << endl;
        return {};
    }

    if (!context->setTensorAddress(outputTensorName.c_str(), dOutput)) {
        cerr << "Failed to set output tensor address." << endl;
        return {};
    }

    if (!context->enqueueV3(stream)) {
        cerr << "TensorRT inference failed." << endl;
        return {};
    }

    err = cudaMemcpyAsync(
        outputData.data(),
        dOutput,
        outputBytes,
        cudaMemcpyDeviceToHost,
        stream
    );
    if (err != cudaSuccess) {
        cerr << "cudaMemcpyAsync output failed: " << cudaGetErrorString(err) << endl;
        return {};
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << endl;
        return {};
    }

    return outputData;
}

vector<Detection> stopSignDetect::detect(const cv::Mat& frame)
{
    vector<Detection> detections;

    vector<float> raw = inferRaw(frame);
    if (raw.empty()) {
        return detections;
    }

    const int numPreds = 8400;
    const float confThreshold = 0.50f;

    float scaleX = static_cast<float>(frame.cols) / static_cast<float>(inputW);
    float scaleY = static_cast<float>(frame.rows) / static_cast<float>(inputH);

    for (int i = 0; i < numPreds; i++) {
        float x = raw[0 * numPreds + i];
        float y = raw[1 * numPreds + i];
        float w = raw[2 * numPreds + i];
        float h = raw[3 * numPreds + i];
        float conf = raw[4 * numPreds + i];

        if (conf < confThreshold) {
            continue;
        }

        int left   = static_cast<int>((x - w / 2.0f) * scaleX);
        int top    = static_cast<int>((y - h / 2.0f) * scaleY);
        int width  = static_cast<int>(w * scaleX);
        int height = static_cast<int>(h * scaleY);

        if (width <= 0 || height <= 0) {
            continue;
        }

        if (left < 0) left = 0;
        if (top < 0) top = 0;
        if (left + width > frame.cols) {
            width = frame.cols - left;
        }
        if (top + height > frame.rows) {
            height = frame.rows - top;
        }

        Detection det;
        det.box = cv::Rect(left, top, width, height);
        det.confidence = conf;
        det.class_id = 0;
        det.label = "STOP";

        detections.push_back(det);
    }

    return detections;
}

vector<float> stopSignDetect::preprocess(const cv::Mat& frame)
{
    cv::Mat resized, rgb, floatImg;

    cv::resize(frame, resized, cv::Size(inputW, inputH));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(floatImg, CV_32F, 1.0 / 255.0);

    vector<float> inputData(static_cast<size_t>(inputC) * inputH * inputW);
    vector<cv::Mat> chw(inputC);

    for (int i = 0; i < inputC; i++) {
        chw[i] = cv::Mat(inputH, inputW, CV_32F,
                         inputData.data() + static_cast<size_t>(i) * inputH * inputW);
    }

    cv::split(floatImg, chw);

    return inputData;
}
