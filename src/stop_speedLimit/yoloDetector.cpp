
#include "yoloDetector.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace std; 

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING) {
        cout << msg << endl;
    }
}


YoloDetector::YoloDetector()
    : YoloDetector(vector<string>{"OBJECT"}, 0.50f, 0.45f)
{
}

YoloDetector::YoloDetector(const vector<string>& classNames,
                           float confThreshold,
                           float nmsThreshold)
    : runtime(nullptr),
      engine(nullptr),
      context(nullptr),
      inputW(0),
      inputH(0),
      inputC(0),
      numClasses_(0),
      numPreds_(0),
      inputBytes(0),
      outputBytes(0),
      dInput(nullptr),
      dOutput(nullptr),
      stream(nullptr),
      classNames_(classNames),
      confThreshold_(confThreshold),
      nmsThreshold_(nmsThreshold)
{
    cout << "yoloDetector created" << endl;
}


YoloDetector::~YoloDetector()
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

    cout << "yoloDetector destroyed" << endl;
}




vector<char> YoloDetector::readEngineFile(const string& enginePath)
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

bool YoloDetector::loadEngine(const string& enginePath)
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

    // yolov8/yolov11 export layout: [1, 4 + numClasses, numPreds]
    // channels 0..3 = box (x,y,w,h), channels 4..(3+C) = per-class scores
    if (outputDims.nbDims >= 3) {
        numClasses_ = outputDims.d[1] - 4;
        numPreds_   = outputDims.d[2];
    } else {
        cerr << "Unexpected output tensor rank: " << outputDims.nbDims << endl;
        return false;
    }

    if (numClasses_ <= 0 || numPreds_ <= 0) {
        cerr << "Invalid derived dims: numClasses=" << numClasses_
             << " numPreds=" << numPreds_ << endl;
        return false;
    }

    // warn if user-provided class name list size doesn't match the engine
    if (!classNames_.empty() && static_cast<int>(classNames_.size()) != numClasses_) {
        cerr << "Warning: classNames size (" << classNames_.size()
             << ") != engine numClasses (" << numClasses_ << ")." << endl;
    }

    cout << "Input tensor name: " << inputTensorName << endl;
    cout << "Output tensor name: " << outputTensorName << endl;
    cout << "numClasses: " << numClasses_ << " | numPreds: " << numPreds_ << endl;
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



vector<float> YoloDetector::inferRaw(const cv::Mat& frame)
{
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

vector<Detection> YoloDetector::detect(const cv::Mat& frame)
{
    vector<Detection> detections;

    vector<float> raw = inferRaw(frame);
    if (raw.empty()) {
        return detections;
    }

    float scaleX = static_cast<float>(frame.cols) / static_cast<float>(inputW);
    float scaleY = static_cast<float>(frame.rows) / static_cast<float>(inputH);

    // pre-nms candidate buffers
    vector<cv::Rect> boxes;
    vector<float>    scores;
    vector<int>      classIds;
    boxes.reserve(64);
    scores.reserve(64);
    classIds.reserve(64);

    for (int i = 0; i < numPreds_; i++) {
        float x = raw[0 * numPreds_ + i];
        float y = raw[1 * numPreds_ + i];
        float w = raw[2 * numPreds_ + i];
        float h = raw[3 * numPreds_ + i];

        // argmax over class scores (channels 4..(3+numClasses_))
        float bestScore = 0.0f;
        int   bestClass = -1;
        for (int c = 0; c < numClasses_; c++) {
            float s = raw[(4 + c) * numPreds_ + i];
            if (s > bestScore) {
                bestScore = s;
                bestClass = c;
            }
        }

        if (bestClass < 0 || bestScore < confThreshold_) {
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
        if (top  < 0) top  = 0;
        if (left + width  > frame.cols) width  = frame.cols - left;
        if (top  + height > frame.rows) height = frame.rows - top;

        boxes.emplace_back(left, top, width, height);
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    if (boxes.empty()) {
        return detections;
    }

    // class-agnostic nms keeps it simple; switch to per-class if you see overlaps within a class
    vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold_, nmsThreshold_, keep);

    detections.reserve(keep.size());
    for (int idx : keep) {
        Detection det;
        det.box        = boxes[idx];
        det.confidence = scores[idx];
        det.class_id   = classIds[idx];

        if (classIds[idx] >= 0
            && classIds[idx] < static_cast<int>(classNames_.size()))
        {
            det.label = classNames_[classIds[idx]];
        } else {
            det.label = "class_" + to_string(classIds[idx]);
        }

        detections.push_back(det);
    }

    return detections;
}

vector<float> YoloDetector::preprocess(const cv::Mat& frame)
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
