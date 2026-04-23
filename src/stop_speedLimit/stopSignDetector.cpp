
#include "stopSignDetector.h"
#include <iostream>

using namespace std; 

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING) {
        cout << msg << endl;
    }
}


stopSignDetect::stopSignDetect()
{
    cout << "stopSignDetect created" << endl;
}

stopSignDetect::~stopSignDetect()
{
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

    for (int i = 0; i < numIOTensors; i++) {
        const char* tensorName = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(tensorName);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(tensorName);

        cout << "Tensor " << i << ": " << tensorName << " | ";
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            cout << "INPUT";
            inputTensorName = tensorName;
        } else {
            cout << "OUTPUT";
            outputTensorName = tensorName;
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

    cout << "Input tensor name: " << inputTensorName << endl;
    cout << "Output tensor name: " << outputTensorName << endl;

    return true;
}



vector<float> stopSignDetect::inferRaw(const cv::Mat& frame) //raw ouput 
{
    cout << "inferRaw called frame size" << frame.cols << "x" <<frame.rows << endl;
    
    vector <float> output; 
    output.push_back(1.0f);
    output.push_back(2.0f);
    output.push_back(3.0f);

    return output; 
}


svector<Detection> stopSignDetect::detect(const cv::Mat& frame)
{

    vector<Detection> detecs;
    return detecs; 

}

