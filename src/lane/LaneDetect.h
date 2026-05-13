#pragma once

#ifndef LANE_DETECT_H
#define LANE_DETECT_H

#include <vector>
#include <limits>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

enum class LaneDetectMode {
    HoughLines,
    SlidingWindow
};

struct ROIConfig {
    double yTop;
    double yBottom;
    double topLeftX;
    double topRightX;
    double bottomLeftX;
    double bottomRightX;
};

struct LaneSmoother {
    bool hasPrev = false;

    cv::Vec3d leftFit;
    cv::Vec3d rightFit;

    double alpha = 0.85;
    int missedFrames = 0;
    int maxMissedFrames = 5;
};

struct ROI {
    cv::Mat mask;
    std::array<cv::Point, 4> corners;
};

// struct TransformResults {
//     cv::Mat warped; // transformed image
//     cv::Mat M;      // trapezoid -> rectangle
//     cv::Mat invM;   // inverse transform
// };

struct LaneDetectionResult {
    ROI roi;
    cv::Mat outputFrame;
    cv::Mat warpedBinary;
    cv::Vec3d leftFit;
    cv::Vec3d rightFit;
};

class LaneDetect {
public:
    explicit LaneDetect(const ROIConfig& roiMod): roiMod(roiMod) {}

    cv::Mat runHough(const cv::Mat& src);
    ROI getROI(const cv::Mat& src, ROIConfig roiMod) const;
    LaneDetectionResult runSlidingWindow(const cv::Mat& frame);

    bool debug = false;

private:
    ROIConfig roiMod;
    bool transformMatrixComputed = false;

    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::Vec3d prevRightFit = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d prevLeftFit = cv::Vec3d(0.0, 0.0, 0.0);
    bool hasPrevFits = false;

    cv::Mat transformMatrix;
    cv::Mat inverseTransformMatrix;

    bool isFitZero(const cv::Vec3d& fit) const;
    float computeSlope(const cv::Vec4i& line) const;
    float lineMidpointX(const cv::Vec4i& line) const;
    bool isValidLeftLine(const cv::Vec4i& line, int imageWidth) const;
    bool isValidRightLine(const cv::Vec4i& line, int imageWidth) const;
    ROI processFrame(const cv::Mat& matSrc, const LaneDetectMode mode) const;

    cv::Mat colorThreshold(const cv::Mat& src) const;
    cv::Mat edgeThreshold(const cv::Mat& src, const LaneDetectMode mode) const;

    cv::Mat birdsEyeTransform(
        const ROI& roi,
        cv::Size& dstSize);

    cv::Mat generateHistogram(const cv::Mat& warped) const;
    std::pair<int, int> getHistogramPeaks(const cv::Mat& histogram) const;

    std::pair<cv::Vec3d, cv::Vec3d> slidingWindowSearch(const cv::Mat& src) const;
    std::pair<cv::Vec3d, cv::Vec3d> previousWindowSearch(
        const cv::Mat& binary,
        const cv::Vec3d& prevLeftFit,
        const cv::Vec3d& prevRightFit) const;

    cv::Vec3d fitPolynomial(const std::vector<cv::Point>& points) const;

    std::string getTurnDirection(
        const cv::Vec3d& leftFit,
        const cv::Vec3d& rightFit) const;

    cv::Mat drawOverlay(
        const cv::Mat& src,
        const cv::Vec3d& leftFit,
        const cv::Vec3d& rightFit,
        const std::string& direction,
        const cv::Size& warpedSize) const;

    cv::Mat drawROIOverlay(
        const cv::Mat& frame,
        const ROI& roi) const;

    bool checkFitValidity(
        const cv::Vec3d& leftFit,
        const cv::Vec3d& rightFit,
        int height) const;
};

#endif // LANE_DETECT_H