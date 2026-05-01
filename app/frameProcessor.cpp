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
                                          ped::PedestrianDetector& pedDetector,
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
        cv::putText(results.finalFrame, text,
                    cv::Point(det.box.x, det.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    // draw speed detections in yellow
    for (const auto& det : results.speedDetections) {
        cv::rectangle(results.finalFrame, det.box, cv::Scalar(0, 255, 255), 2);
        std::string text = det.label + " " + std::to_string(det.confidence).substr(0, 4);
        cv::putText(results.finalFrame, text,
                    cv::Point(det.box.x, det.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }

    // pedestrian detection + risk visualization
    results.pedDetections = pedDetector.detect(results.finalFrame);
    pedDetector.visualize(results.finalFrame, results.pedDetections);

    // pedestrian HUD
    int n_high = 0, n_med = 0, n_low = 0;
    for (const auto& p : results.pedDetections) {
        if      (p.risk == ped::RiskLevel::HIGH)   n_high++;
        else if (p.risk == ped::RiskLevel::MEDIUM)  n_med++;
        else                                        n_low++;
    }
    if (!results.pedDetections.empty()) {
        cv::putText(results.finalFrame,
            "Peds: " + std::to_string(results.pedDetections.size()) +
            "  H:" + std::to_string(n_high) +
            " M:" + std::to_string(n_med) +
            " L:" + std::to_string(n_low),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    // warning text
    if (!results.stopDetections.empty()) {
        results.warningText = "Slow down to a stop, stop sign detected";
    } else if (!results.speedDetections.empty()) {
        Detection best = results.speedDetections[0];
        for (const auto& det : results.speedDetections) {
            if (det.confidence > best.confidence) best = det;
        }
        results.warningText = best.label + " detected";
    } else {
        results.warningText = "";
    }

    if (!results.warningText.empty()) {
        cv::putText(results.finalFrame, results.warningText,
                    cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 0, 255), 2);
    }

    return results;
}
