#include "LaneDetect.h"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

#include "opencv2/highgui.hpp"

/**
 * Checks if a fit has failed (is all zeroes)
 */
bool LaneDetect::isFitZero(const cv::Vec3d& fit) const
{
    return fit == cv::Vec3d(0.0, 0.0, 0.0) ? true : false;
}

/**
 * Debug helper to make ROI easy to see
 */
cv::Mat LaneDetect::drawROIOverlay(
    const cv::Mat& frame,
    const ROI& roi) const
{
    cv::Mat output = frame.clone();

    // Translucent fill
    cv::Mat overlay = output.clone();
    cv::fillConvexPoly(overlay, roi.corners, cv::Scalar(0, 255, 0));
    cv::addWeighted(overlay, 0.20, output, 0.80, 0.0, output);

    // Outline
    const cv::Point* pts = roi.corners.data();
    int npts = static_cast<int>(roi.corners.size());
    cv::polylines(output, &pts, &npts, 1, true, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    // Corners
    for (size_t i = 0; i < roi.corners.size(); ++i) {
        cv::circle(output, roi.corners[i], 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }

    return output;
}

/**
 * ---For Hough line version---
 * Calculates the slope for a line
 */
float LaneDetect::computeSlope(const cv::Vec4i& line) const
{
    float dx = static_cast<float>(line[2] - line[0]);
    float dy = static_cast<float>(line[3] - line[1]);

    if (std::abs(dx) < 0.001f) {
        return std::numeric_limits<float>::infinity();
    }

    return dy / dx;
}

/**
 * ---For Hough line version---
 * Returns X value of a line's midpoint coordinate; used to asses if a lane line 
 * has crossed the screen in a way that doesn't make sense
 */
float LaneDetect::lineMidpointX(const cv::Vec4i& line) const
{
    return 0.5f * static_cast<float>(line[0] + line[2]);
}

/**
 * ---For Hough line version---
 * Checks for disqualifying traits in a left-side detected lane line
 */
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

/**
 * ---For Hough line version---
 * Checks for disqualifying traits in a right-side detected lane line
 */
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

/**
 * Returns ROI mask and corner coordinates; adjusted manually
 */
ROI LaneDetect::getROI(const cv::Mat& src, const ROIConfig roiMod) const
{
    int height = src.rows;
    int width = src.cols;

    int yTop = static_cast<int>(height * roiMod.yTop);
    int yBottom = static_cast<int>(height * roiMod.yBottom);

    int topLeftX = static_cast<int>(width * roiMod.topLeftX);
    int topRightX = static_cast<int>(width * roiMod.topRightX);
    int bottomLeftX = static_cast<int>(width * roiMod.bottomLeftX);
    int bottomRightX = static_cast<int>(width * roiMod.bottomRightX);

    std::array<cv::Point, 4> trapezoid = {
        cv::Point(topLeftX, yTop),          // TL
        cv::Point(topRightX, yTop),         // TR
        cv::Point(bottomRightX, yBottom),   // BR
        cv::Point(bottomLeftX, yBottom)     // BL
    };

    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::fillConvexPoly(mask, trapezoid, cv::Scalar(255));

    return {mask, trapezoid};
}

/**
 * Attempts to isolate white an yellow lane lines; returns a mask
 */
cv::Mat LaneDetect::colorThreshold(const cv::Mat& src) const
{
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat whiteMask, yellowMask;

    // Lane color ranges
    cv::Scalar lowerYellow(12, 40, 90);
    cv::Scalar upperYellow(38, 255, 255);
    cv::Scalar lowerWhite(0, 0, 140);
    cv::Scalar upperWhite(179, 60, 255);

    // Create masks from ranges
    cv::inRange(hsv, lowerWhite, upperWhite, whiteMask);
    cv::inRange(hsv, lowerYellow, upperYellow, yellowMask);

    cv::Mat colorMask;
    cv::bitwise_or(whiteMask, yellowMask, colorMask);
    cv::dilate(colorMask, colorMask, LaneDetect::morphKernel);

    if (debug) {
        cv::Mat color = cv::Mat::zeros(src.size(), src.type());
        src.copyTo(color, colorMask);
        imshow("color mask", color);
    }

    return colorMask;
}

/**
 * Performs edge detection and returns a mask
 */
cv::Mat LaneDetect::edgeThreshold(const cv::Mat& src, const LaneDetectMode mode) const
{
    cv::Mat mask, grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    cv::Canny(grey, mask, 50, 150, 3);

    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, LaneDetect::morphKernel);

    return mask;
}

/**
 * Applies an ROI, blurs, thresholds based on color and edge detection, then
 * returns a mask of the result
 */
ROI LaneDetect::processFrame(const cv::Mat& src, const LaneDetectMode mode) const
{
    ROI roi = getROI(src, roiMod);

    // Rectangle based on ROI with additional padding to avoid boundary issues with blue and Canny
    // Reduce processing area without dealing with trapezoid shape
    const int padding = 15;
    std::array<cv::Point, 4> boundCorners;
    for (int i = 0; i < boundCorners.size(); i++) {
        boundCorners[i] = cv::Point(roi.corners[i].x + padding, roi.corners[i].y + padding);
    }
    cv::Rect roiBounds = cv::boundingRect(boundCorners);

    cv::Mat srcCropped = src(roiBounds);
    cv::Mat cropBlurred;

    cv::GaussianBlur(srcCropped, cropBlurred, cv::Size(5, 5), 0);
    
    cv::Mat colorMask = colorThreshold(cropBlurred);
    cv::Mat edgeMask = edgeThreshold(cropBlurred, mode);

    cv::Mat combinedMask;
    cv::bitwise_and(colorMask, edgeMask, combinedMask);

    // Copy mask to full sized frame mask
    cv::Mat combinedMaskFull = cv::Mat::zeros(src.size(), CV_8UC1);
    combinedMask.copyTo(combinedMaskFull(roiBounds));

    cv::bitwise_and(combinedMaskFull, roi.mask, roi.mask);

    return roi;
}

/**
 * Hough line version -- not as worked on as the sliding windows mode
 */
cv::Mat LaneDetect::runHough(const cv::Mat& src)
{
    int rho = 1;
    int thetaDivisor = 180;
    int thresholdProb = 50;
    int minLineLength = 50;
    int maxLineGap = 10;

    cv::Mat matRoi = processFrame(src, LaneDetectMode::HoughLines).mask;

    cv::Mat matColoredSrc = src.clone();

    std::vector<cv::Vec4i> linesP;
    cv::HoughLinesP(
        matRoi,
        linesP,
        rho,
        CV_PI / thetaDivisor,
        thresholdProb,
        minLineLength,
        maxLineGap
    );

    for (size_t i = 0; i < linesP.size(); ++i) {
        cv::Vec4i l = linesP[i];

        if (!isValidLeftLine(l, src.cols) && !isValidRightLine(l, src.cols)) {
            continue;
        }

        cv::Point pt1(l[0], l[1]);
        cv::Point pt2(l[2], l[3]);

        cv::line(matColoredSrc, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    return matColoredSrc;
}

cv::Mat LaneDetect::birdsEyeTransform(
    const ROI& roi,
    cv::Size& dstSize)
{
    std::array<cv::Point2f, 4> src {
        cv::Point2f(static_cast<float>(roi.corners[0].x), static_cast<float>(roi.corners[0].y)), // TL
        cv::Point2f(static_cast<float>(roi.corners[1].x), static_cast<float>(roi.corners[1].y)), // TR
        cv::Point2f(static_cast<float>(roi.corners[2].x), static_cast<float>(roi.corners[2].y)), // BR
        cv::Point2f(static_cast<float>(roi.corners[3].x), static_cast<float>(roi.corners[3].y))  // BL
    };

    // Bottom width of ROI trapezoid (BR.x - BL.x)
    float bottomWidth = src[2].x - src[3].x;

    float dstHeight = (float)(dstSize.height - 1);
    float dstWidth = (float)(dstSize.width - 1);

    std::array<cv::Point2f, 4> dst {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(dstWidth, 0.0f),
        cv::Point2f(dstWidth, dstHeight),
        cv::Point2f(0.0f, dstHeight),
    };

    if (!transformMatrixComputed) {
        transformMatrix = cv::getPerspectiveTransform(src, dst);
        inverseTransformMatrix = cv::getPerspectiveTransform(dst, src);
        transformMatrixComputed = true;
    }   

    cv::Mat warped;
    cv::warpPerspective(roi.mask, warped, transformMatrix, dstSize, cv::INTER_NEAREST);

    return warped;
}

cv::Mat LaneDetect::generateHistogram(const cv::Mat& binary) const
{
    int height = binary.rows;
    int width  = binary.cols;

    cv::Mat bottomHalf = binary.rowRange(height / 2, height);

    cv::Mat histogram;
    cv::reduce(bottomHalf, histogram, 0, cv::REDUCE_SUM, CV_32S);

    return histogram;
}

std::pair<int, int> LaneDetect::getHistogramPeaks(const cv::Mat& histogram) const
{
    int midpoint = histogram.cols / 2;
    int offset = 10; // offset to try to prevent detection of side-by-side markings

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    // Left max
    cv::Mat leftHalf = histogram.colRange(0, midpoint + offset);
    cv::minMaxLoc(leftHalf, &minVal, &maxVal, &minLoc, &maxLoc);
    int leftPeakX = maxLoc.x;

    // Right max
    cv::Mat rightHalf = histogram.colRange(midpoint + offset, histogram.cols);
    cv::minMaxLoc(rightHalf, &minVal, &maxVal, &minLoc, &maxLoc);
    int rightPeakX = maxLoc.x + midpoint + offset;

    return {leftPeakX, rightPeakX};
}

/**
 * Returns the coefficients of a second polynomial fit 
 * for f(y) = A*y^2 + B*y + C
 * 
 * Uses weighted least squares to give more credence to the base pixels
 * ChatGPT made this because I don't trust myself to code the math correctly -- thanks Chat!
 */
cv::Vec3d LaneDetect::fitPolynomial(const std::vector<cv::Point>& points) const
{
    const int n = static_cast<int>(points.size());

    if (n < 3) {
        return cv::Vec3d(0.0, 0.0, 0.0);
    }

    double maxY = 0.0;
    for (const cv::Point& p : points) {
        maxY = std::max(maxY, static_cast<double>(p.y));
    }

    double sumW   = 0.0;
    double sumWY  = 0.0;
    double sumWY2 = 0.0;
    double sumWY3 = 0.0;
    double sumWY4 = 0.0;

    double sumWX   = 0.0;
    double sumWXY  = 0.0;
    double sumWXY2 = 0.0;

    for (const cv::Point& p : points) {
        const double y = static_cast<double>(p.y);
        const double x = static_cast<double>(p.x);

        // 0.25 near the top, 1.0 near the base.
        const double normalizedY = maxY > 0.0 ? y / maxY : 1.0;
        const double w = 0.25 + 0.75 * normalizedY * normalizedY;

        const double y2 = y * y;
        const double y3 = y2 * y;
        const double y4 = y2 * y2;

        sumW   += w;
        sumWY  += w * y;
        sumWY2 += w * y2;
        sumWY3 += w * y3;
        sumWY4 += w * y4;

        sumWX   += w * x;
        sumWXY  += w * x * y;
        sumWXY2 += w * x * y2;
    }

    // Weighted normal equations for:
    // x = A*y^2 + B*y + C
    cv::Matx33d lhs(
        sumWY4, sumWY3, sumWY2,
        sumWY3, sumWY2, sumWY,
        sumWY2, sumWY,  sumW
    );

    cv::Vec3d rhs(sumWXY2, sumWXY, sumWX);

    cv::Vec3d beta;
    const bool solved = cv::solve(lhs, rhs, beta, cv::DECOMP_CHOLESKY);

    if (!solved) {
        return cv::Vec3d(0.0, 0.0, 0.0);
    }

    return beta; // A, B, C
}

/**
 * Checks if a fit seems reasonable;
 * any cases where the left line and right line are crossed should be rejected
 * and a minimum gap should be maintained
 */
bool LaneDetect::checkFitValidity(
    const cv::Vec3d& leftFit,
    const cv::Vec3d& rightFit,
    int height) const
{
    // Zero fit indicates invalid
    if (isFitZero(leftFit) || isFitZero(rightFit)) {
        return false;
    }   
    
    for (int y = 0; y < height; y += 10) {
        double lx = leftFit[0] * y * y + leftFit[1] * y + leftFit[2];
        double rx = rightFit[0] * y * y + rightFit[1] * y + rightFit[2];

        // Right lane X values should be greater than left lane
        if (rx <= lx) {
            return false;
        }
        // Enforce a minimum gap
        if ((rx - lx) < 40.0) {
            return false;
        }
    }

    return true;
}

/**
 * Slides windows along a warped binary image to find active pixels
 * Based off a function from:
 * https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/
 */
std::pair<cv::Vec3d, cv::Vec3d> LaneDetect::slidingWindowSearch(const cv::Mat& binary) const
{
    int numWindows = 10;
    int margin = binary.cols * 0.075;
    int recenterThreshold = 75;
    int windowHeight = binary.rows / numWindows;

    cv::Mat windowsMat;

    // Create histogram and starting X loc for windows
    cv::Mat histogram = generateHistogram(binary);
    auto [leftBaseX, rightBaseX] = getHistogramPeaks(histogram);

    // Find all non-zero pixels in the frame
    std::vector<cv::Point> nonZeroPts;
    cv::findNonZero(binary, nonZeroPts);

    std::vector<std::vector<cv::Point>> pointsByWindow(numWindows);

    for (const cv::Point& pt : nonZeroPts) {
        int band = (binary.rows - 1 - pt.y) / windowHeight;

        if (band >= 0 && band < numWindows) {
            pointsByWindow[band].push_back(pt);
        }
    }

    if (debug) {
        // Conversion for drawing windows in color
        cv::cvtColor(binary, windowsMat, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Point> leftPts;
    std::vector<cv::Point> rightPts;
    leftPts.reserve(nonZeroPts.size() / 2);
    rightPts.reserve(nonZeroPts.size() / 2);

    for (int i = 0; i < numWindows; i++) {
        int winYLow = binary.rows - (i + 1) * windowHeight;
        int winYHigh = binary.rows - i * windowHeight;
        int winXLeftLow = leftBaseX - margin;
        int winXLeftHigh = leftBaseX + margin;
        int winXRightLow = rightBaseX - margin;
        int winXRightHigh = rightBaseX + margin;

        if (debug) {
            // Draw windows
            cv::rectangle(
                windowsMat,
                cv::Point(winXLeftLow, winYLow),
                cv::Point(winXLeftHigh, winYHigh),
                cv::Scalar(0, 255, 255),
                2
            );

            cv::rectangle(
                windowsMat,
                cv::Point(winXRightLow, winYLow),
                cv::Point(winXRightHigh, winYHigh),
                cv::Scalar(0, 255, 255),
                2
            );
        }

        int goodLeftCount = 0;
        int goodRightCount = 0;
        int goodLeftSumX = 0;
        int goodRightSumX = 0;

        int divider = (leftBaseX + rightBaseX) / 2;
        const std::vector<cv::Point>& bandPts = pointsByWindow[i];

        for (const cv::Point& pt : bandPts) {
            if (pt.x >= winXLeftLow && pt.x < winXLeftHigh && pt.x < divider) {
                leftPts.push_back(pt);
                goodLeftSumX += pt.x;
                goodLeftCount++;
            } 
            else if (pt.x >= winXRightLow && pt.x < winXRightHigh && pt.x >= divider) {
                rightPts.push_back(pt);
                goodRightSumX += pt.x;
                goodRightCount++;
            }
        }

        if (goodLeftCount > recenterThreshold) {
            leftBaseX = goodLeftSumX / goodLeftCount;
        }

        if (goodRightCount > recenterThreshold) {
            rightBaseX = goodRightSumX / goodRightCount;
        }
    }

    cv::Vec3d leftFit = fitPolynomial(leftPts);
    cv::Vec3d rightFit = fitPolynomial(rightPts);

    if (debug) {
        imshow("sliding windows", windowsMat);
    }

    return {leftFit, rightFit};
}

/**
 * Attempts to fit previously calculated polynomial during search;
 * should minimize number of pixels checked
 * Based off a function from: https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/
 */
std::pair<cv::Vec3d, cv::Vec3d> LaneDetect::previousWindowSearch(
    const cv::Mat& binary,
    const cv::Vec3d& prevLeftFit,
    const cv::Vec3d& prevRightFit) const
{
    int margin = static_cast<int>(binary.cols * 0.075);

    std::vector<cv::Point> nonZeroPts;
    cv::findNonZero(binary, nonZeroPts);

    std::vector<cv::Point> leftPts;
    std::vector<cv::Point> rightPts;

    leftPts.reserve(nonZeroPts.size() / 2);
    rightPts.reserve(nonZeroPts.size() / 2);

    for (const cv::Point& pt : nonZeroPts) {
        double y = static_cast<double>(pt.y);

        double expectedLeftX =
            prevLeftFit[0] * y * y +
            prevLeftFit[1] * y +
            prevLeftFit[2];

        double expectedRightX =
            prevRightFit[0] * y * y +
            prevRightFit[1] * y +
            prevRightFit[2];

        if (std::abs(pt.x - expectedLeftX) <= margin) {
            leftPts.push_back(pt);
        }

        if (std::abs(pt.x - expectedRightX) <= margin) {
            rightPts.push_back(pt);
        }
    }

    cv::Vec3d leftFit = fitPolynomial(leftPts);
    cv::Vec3d rightFit = fitPolynomial(rightPts);

    if (debug) {
        cv::Mat debugMat;
        cv::cvtColor(binary, debugMat, cv::COLOR_GRAY2BGR);
        cv::imshow("previous window search", debugMat);
    }

    return {leftFit, rightFit};
}

/**
 * Draws overlay for area between lanes and displays text for lane direction
 */
cv::Mat LaneDetect::drawOverlay(
    const cv::Mat& src,
    const cv::Vec3d& leftFit,
    const cv::Vec3d& rightFit,
    const std::string& direction,
    const cv::Size& warpedSize) const
{
    std::vector<cv::Point> leftPts;
    std::vector<cv::Point> rightPts;

    for (int y = 0; y < warpedSize.height; y += 10) {
        double lx = (leftFit[0] * y * y) + (leftFit[1] * y) + leftFit[2];
        double rx = (rightFit[0] * y * y) + (rightFit[1] * y) + rightFit[2];

        cv::Point leftPt(static_cast<int>(lx), y);
        cv::Point rightPt(static_cast<int>(rx), y);

        leftPts.push_back(leftPt);
        rightPts.push_back(rightPt);
    }

    std::reverse(rightPts.begin(), rightPts.end());

    std::vector<cv::Point> lanePts;
    lanePts.insert(lanePts.end(), leftPts.begin(), leftPts.end());
    lanePts.insert(lanePts.end(), rightPts.begin(), rightPts.end());

    cv::Mat laneWarped = cv::Mat::zeros(warpedSize, CV_8UC3);
    cv::fillPoly(laneWarped, std::vector<std::vector<cv::Point>>{lanePts}, cv::Scalar(0, 255, 0));

    // Draws differently colored left and right lines
    if (debug) {
        cv::polylines(laneWarped, leftPts, false, cv::Scalar(255, 0, 0), 10);
        cv::polylines(laneWarped, rightPts, false, cv::Scalar(0, 0, 255), 10);
    }

    cv::Mat laneUnwarped;
    cv::warpPerspective(laneWarped, laneUnwarped, inverseTransformMatrix, src.size());

    cv::Mat output;
    cv::addWeighted(src, 1.0, laneUnwarped, 0.35, 0.0, output);

    cv::putText(
        output, 
        direction, 
        cv::Point(output.cols * 0.83, output.rows * 0.08), 
        cv::FONT_HERSHEY_PLAIN, 
        3.0, 
        cv::Scalar(0, 0, 255),
        4
    );

    return output;
}

/**
 * Calls necessary functions to perform sliding window lane detection
 */
LaneDetectionResult LaneDetect::runSlidingWindow(const cv::Mat& frame)
{
    LaneDetectionResult result;

    result.roi = processFrame(frame, LaneDetectMode::SlidingWindow);

    if (debug) {
        cv::Mat roiPreview = drawROIOverlay(frame, result.roi);
        cv::imshow("ROI overlay", roiPreview);
    }

    cv::Size frameSize(frame.cols, frame.rows);

    result.warpedBinary = birdsEyeTransform(result.roi, frameSize);

    cv::Vec3d leftFit, rightFit;
    if (hasPrevFits) {
        std::tie(leftFit, rightFit) =
            previousWindowSearch(result.warpedBinary, prevLeftFit, prevRightFit);

        if (!checkFitValidity(leftFit, rightFit, result.warpedBinary.rows)) {
            std::tie(leftFit, rightFit) = slidingWindowSearch(result.warpedBinary);
        }
    } 
    else {
        std::tie(leftFit, rightFit) = slidingWindowSearch(result.warpedBinary);
    }

    if (!checkFitValidity(leftFit, rightFit, result.warpedBinary.rows)) {
        leftFit = cv::Vec3d(0.0, 0.0, 0.0);
        rightFit = cv::Vec3d(0.0, 0.0, 0.0);

        result.outputFrame = frame.clone();

        cv::putText(
            result.outputFrame, 
            "Unknown", 
            cv::Point(result.outputFrame.cols * 0.83, result.outputFrame.rows * 0.08), 
            cv::FONT_HERSHEY_PLAIN, 
            2.6, 
            cv::Scalar(0, 0, 255),
            4
        );

        return result;
    }
   
    prevLeftFit = leftFit;
    prevRightFit = rightFit;
    hasPrevFits = true;

    std::string direction = getTurnDirection(leftFit, rightFit);

    result.outputFrame = drawOverlay(
        frame, 
        leftFit, 
        rightFit, 
        direction, 
        result.warpedBinary.size()
    );

    return result;
}

/**
 * Uses the A coefficient to check for lane curvature
 * 
 * This is probably not entirely robust since B can effect curvature too
 */
std::string LaneDetect::getTurnDirection(
    const cv::Vec3d& leftFit,
    const cv::Vec3d& rightFit) const
{
    // Used a graphing calculator to see what looked reasonable
    float curvatureThreshold = 0.00015;

    bool leftValid = isFitZero(leftFit) ? false : true;
    bool rightValid = isFitZero(rightFit) ? false : true;

    if (!leftValid && !rightValid) {
        return "Unknown";
    }

    double A = 0.0;

    if (leftValid && rightValid) {
        A = 0.5 * (leftFit[0] + rightFit[0]);
    } 
    else if (leftValid) {
        A = leftFit[0];
    } 
    else {
        A = rightFit[0];
    }

    if (A > curvatureThreshold) {
        return "Right";
    }

    if (A < -curvatureThreshold) {
        return "Left";
    }

    return "Straight";
}