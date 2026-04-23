
#ifndef STOP_DETECT
#define STOP_DETECT 

#include <string>
#include<vector>
#include<opencv2/core.hpp> 

struct Detection {
    cv::Rect box; 
    float confidence;
    int class_id;
    std::string label; 
};

class stopSignDetect{

    public: 
    stopSignDetect();  //cons
    ~stopSignDetect(); //dest 

    bool loadEngine(const std::string& enginePath); //loads the model engine 
    std::vector<float> inferRaw(const cv::Mat& frame); //raw model output 
    std::vector<Detection> detect(const cv::Mat& frame); //


    private:

};


#endif //STOP_DETECT
