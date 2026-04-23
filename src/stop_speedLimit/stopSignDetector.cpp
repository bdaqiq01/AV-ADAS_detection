
#include "stopSignDetector.h"
#include <iostream>

using namespace std; 


stopSignDetect::stopSignDetect()
{
    cout << "stopSignDetect created" << endl;
}

stopSignDetect::~stopSignDetect()
{
    cout << "stopSignDetect destroyed" << endl;
}


bool stopSignDetect::loadEngine(const std::string& enginePath) //loads the model engine 
{
    cout << "loading engine " <<endl; 
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

