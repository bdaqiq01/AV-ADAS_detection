#include "LaneDetect.h"

#include <cmath>
#include "opencv2/highgui.hpp"

float LaneDetect::computeSlope(const cv::Vec4i& line) const
{
    float dx = static_cast<float>(line[2] - line[0]);
    float dy = static_cast<float>(line[3] - line[1]);

    if (std::abs(dx) < 0.001f) {
        return std::numeric_limits<float>::infinity();
    }

    return dy / dx;
}

float LaneDetect::lineMidpointX(const cv::Vec4i& line) const
{
    return 0.5f * static_cast<float>(line[0] + line[2]);
}

bool LaneDetect::isValidLeftLine(const cv::Vec4i& line, int imageWidth) const
{
    float slope = computeSlope(line);

    if (std::abs(slope) < 0.3f) {
        return false;
    }

    if (slope >= 0.0f) {
        return false;
    }

    float midX = lineMidpointX(line);
    if (midX > static_cast<float>(imageWidth) * 0.45f) {
        return false;
    }

    return true;
}

bool LaneDetect::isValidRightLine(const cv::Vec4i& line, int imageWidth) const
{
    float slope = computeSlope(line);

    if (std::abs(slope) < 0.3f) {
        return false;
    }

    if (slope <= 0.0f) {
        return false;
    }

    float midX = lineMidpointX(line);
    if (midX < static_cast<float>(imageWidth) * 0.55f) {
        return false;
    }

    return true;
}

cv::Mat LaneDetect::getROI(const cv::Mat& matSrc) const
{
    int height = matSrc.rows;
    int width = matSrc.cols;

    int yTop = static_cast<int>(height * 0.575);
    int yBottom = static_cast<int>(height * 0.8);

    int topLeftX = static_cast<int>(width * 0.35);
    int topRightX = static_cast<int>(width * 0.65);
    int bottomLeftX = 0;
    int bottomRightX = width - 1;

    std::vector<cv::Point> trapezoid = {
        cv::Point(topLeftX, yTop),
        cv::Point(topRightX, yTop),
        cv::Point(bottomRightX, yBottom),
        cv::Point(bottomLeftX, yBottom)
    };

    cv::Mat mask = cv::Mat::zeros(matSrc.size(), CV_8UC1);
    cv::fillConvexPoly(mask, trapezoid, cv::Scalar(255));

    return mask;
}

cv::Mat LaneDetect::processFrame(const cv::Mat& matSrc) const
{
    cv::Mat matDst;
    cv::Mat matBlurred;

    cv::GaussianBlur(matSrc, matBlurred, cv::Size(5, 5), 0);
    cv::Canny(matBlurred, matDst, 50, 150, 3);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(matDst, matDst, cv::MORPH_CLOSE, kernel);

    return matDst;
}

cv::Mat LaneDetect::runHough(const cv::Mat& matSrc, const HoughParams& params)
{
    cv::Mat matDst = processFrame(matSrc);

    cv::Mat matRoiMask = getROI(matDst);
    cv::Mat matRoi;
    cv::bitwise_and(matDst, matDst, matRoi, matRoiMask);

    cv::Mat matDisplayRoi;
    cv::cvtColor(matRoi, matDisplayRoi, cv::COLOR_GRAY2BGR);
   // cv::imshow("roi", matDisplayRoi);

    cv::Mat matColoredSrc = matSrc.clone();

    std::vector<cv::Vec4i> linesP;
    cv::HoughLinesP(
        matRoi,
        linesP,
        params.rho,
        CV_PI / params.thetaDivisor,
        params.thresholdProb,
        params.minLineLength,
        params.maxLineGap
    );

    for (size_t i = 0; i < linesP.size(); ++i) {
        cv::Vec4i l = linesP[i];

        if (!isValidLeftLine(l, matSrc.cols) && !isValidRightLine(l, matSrc.cols)) {
            continue;
        }

        cv::Point pt1(l[0], l[1]);
        cv::Point pt2(l[2], l[3]);

        cv::line(matColoredSrc, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    return matColoredSrc;
}
