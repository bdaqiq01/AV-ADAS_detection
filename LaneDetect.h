#ifndef LANE_DETECT_H
#define LANE_DETECT_H

#include <vector>
#include <limits>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

struct HoughParams {
    int rho;
    int thetaDivisor;
    int thresholdProb;
    int minLineLength;
    int maxLineGap;
};

class LaneDetect {
public:
    LaneDetect() = default;
    ~LaneDetect() = default;

    cv::Mat runHough(const cv::Mat& matSrc, const HoughParams& params);
    cv::Mat getROI(const cv::Mat& matSrc) const;

private:
    float computeSlope(const cv::Vec4i& line) const;
    float lineMidpointX(const cv::Vec4i& line) const;

    bool isValidLeftLine(const cv::Vec4i& line, int imageWidth) const;
    bool isValidRightLine(const cv::Vec4i& line, int imageWidth) const;

    cv::Mat processFrame(const cv::Mat& matSrc) const;
};

#endif // LANE_DETECT_H
