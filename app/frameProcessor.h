#ifndef FRAME_PROCESSOR_H
#define FRAME_PROCESSOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "LaneDetect.h"
#include "yoloDetector.h"
#include "LaneDetect.h"
#include "PedestrianDetector.h"

struct FrameResults {
    cv::Mat finalFrame;
    std::vector<Detection> stopDetections;
    std::vector<Detection> speedDetections;
    std::vector<ped::TrackedPedestrian> pedDetections;
    std::string warningText;
};

struct FrameProcessorOptions {
    bool disableLane = false;
    bool disableSign = false;
    bool disablePedestrian = false;
};

class FrameProcessor {
public:
    explicit FrameProcessor(const FrameProcessorOptions& options = {});

    ~FrameProcessor() = default;
    FrameResults processFrame(const cv::Mat& frame,
                              LaneDetect& laneDetect,
                              LaneDetectMode &laneMode,
                              YoloDetector& stopDetector,
                              YoloDetector& speedDetector,
                              ped::PedestrianDetector& pedDetector);
private:
    FrameProcessorOptions options_;
    std::string stableWarning_;
    std::string candidateWarning_;
    int candidateCount_;
    int stableFramesRequired_;
    int frameCounter_ = 0;

    std::vector<Detection> cachedStopDetections_;
    std::vector<Detection> cachedSpeedDetections_;
    std::vector<ped::TrackedPedestrian> cachedPedDetections_;
};

#endif
