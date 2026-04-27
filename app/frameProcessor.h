#ifndef FRAME_PROCESSOR_H
#define FRAME_PROCESSOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "LaneDetect.h"
#include "yoloDetector.h"

struct FrameResults {
    cv::Mat finalFrame;
    std::vector<Detection> stopDetections;
    std::vector<Detection> speedDetections;
    std::string warningText;
};

class FrameProcessor {
public:
    FrameProcessor();
    ~FrameProcessor() = default;
    FrameResults processFrame(const cv::Mat& frame,
                              LaneDetect& laneDetect,
                              YoloDetector& stopDetector,
                              YoloDetector& speedDetector,
                              const HoughParams& params);
private:
    std::string stableWarning_;
    std::string candidateWarning_;
    int candidateCount_;
    int stableFramesRequired_;
};

#endif