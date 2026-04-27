#include "frameProcessor.h"

#include <string>
#include <opencv2/imgproc.hpp>

FrameProcessor::FrameProcessor()
    : stableWarning_(""),
      candidateWarning_(""),
      candidateCount_(0),
      stableFramesRequired_(3)
{
}

FrameResults FrameProcessor::processFrame(const cv::Mat& frame,
                                          LaneDetect& laneDetect,
                                          YoloDetector& stopDetector,
                                          YoloDetector& speedDetector,
                                          const HoughParams& params)
{
    FrameResults results;

    // lane detection
    results.finalFrame = laneDetect.runHough(frame, params);

    // stop sign detection
    results.stopDetections = stopDetector.detect(frame);
    
    // speed limit detection
    results.speedDetections = speedDetector.detect(frame);

    // draw stop detections in green
    for (const auto& det : results.stopDetections) {
        cv::rectangle(results.finalFrame, det.box, cv::Scalar(0, 255, 0), 2);

        std::string text = det.label + " " + std::to_string(det.confidence).substr(0, 4);
        cv::putText(results.finalFrame,
                    text,
                    cv::Point(det.box.x, det.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2);
    }

    // draw speed detections in yellow
    for (const auto& det : results.speedDetections) {
        cv::rectangle(results.finalFrame, det.box, cv::Scalar(0, 255, 255), 2);

        std::string text = det.label + " " + std::to_string(det.confidence).substr(0, 4);
        cv::putText(results.finalFrame,
                    text,
                    cv::Point(det.box.x, det.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 255),
                    2);
    }

    // warning text
    if (!results.stopDetections.empty()) {
        results.warningText = "Slow down to a stop, stop sign detected";
    }
    else if (!results.speedDetections.empty()) {
        Detection best = results.speedDetections[0];
        for (const auto& det : results.speedDetections) {
            if (det.confidence > best.confidence) {
                best = det;
            }
        }
        results.warningText = best.label + " detected";
    }
    else {
        results.warningText = "";
    }
    if (!results.warningText.empty()) {
        cv::putText(results.finalFrame,
                    results.warningText,
                    cv::Point(30, 80),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.9,
                    cv::Scalar(0, 0, 255),
                    2);
    }

    return results;
}