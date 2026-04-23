
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

vector<float> stopSignDetect::inferRaw(const cv::Mat& frame) //raw ouput 
{
    cout << "inferRaw called frame size" << frame.cols << "x" <<frame.rows << endl;
    
    vector <float> output; 
    output.push_back(1.0f);
    output.push_back(2.0f);
    output.push_back(3.0f);

    return output; 
}


vector<Detection> stopSignDetect::detect(const cv::Mat& frame)
{

    vector<Detection> detecs;
    return detecs; 

}

