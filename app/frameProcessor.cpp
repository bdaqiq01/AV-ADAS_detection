#include "frameProcessor.h"

#include <string>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <opencv2/imgproc.hpp>

struct LaneThreadData {
    const cv::Mat* frame;
    LaneDetect* laneDetect;
    const HoughParams* params;
    cv::Mat output;
};

struct StopThreadStruct {
    const cv::Mat* frame;
    YoloDetector* detector;
    std::vector<Detection> detections;
};

struct SpeedThreadStruct {
    const cv::Mat* frame;
    YoloDetector* detector;
    std::vector<Detection> detections;
};

static void* laneWorker(void* arg) {
    auto* d = static_cast<LaneThreadData*>(arg);
    d->output = d->laneDetect->runHough(*d->frame, *d->params);
    return nullptr;
}

static void* stopWorker(void* arg) {
    auto* d = static_cast<StopThreadStruct*>(arg);
    d->detections = d->detector->detect(*d->frame);
    return nullptr;
}

static void* speedWorker(void* arg) {
    auto* d = static_cast<SpeedThreadStruct*>(arg);
    d->detections = d->detector->detect(*d->frame);
    return nullptr;
}

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

    // ── Parallel: lane + stop + speed run simultaneously ─────────────────
    LaneThreadData    laneData{&frame, &laneDetect,   &params, {}};
    StopThreadStruct  stopData{&frame, &stopDetector,  {}};
    SpeedThreadStruct speedData{&frame, &speedDetector, {}};

    pthread_t laneThread, stopThread, speedThread;
    pthread_create(&laneThread,  nullptr, laneWorker,  &laneData);
    pthread_create(&stopThread,  nullptr, stopWorker,  &stopData);
    pthread_create(&speedThread, nullptr, speedWorker, &speedData);

    pthread_join(laneThread,  nullptr);
    pthread_join(stopThread,  nullptr);
    pthread_join(speedThread, nullptr);

    results.finalFrame      = std::move(laneData.output);
    results.stopDetections  = std::move(stopData.detections);
    results.speedDetections = std::move(speedData.detections);

    // ── Sequential: pedestrian detection after threads join ───────────────
    // TensorRT contexts are not thread-safe — must run sequentially
    results.pedDetections = pedDetector.detect(results.finalFrame);
    pedDetector.visualize(results.finalFrame, results.pedDetections);

    // ── Draw stop detections in green ─────────────────────────────────────
    for (const auto& det : results.stopDetections) {
        cv::rectangle(results.finalFrame, det.box, cv::Scalar(0, 255, 0), 2);
        std::string text = det.label + " " + std::to_string(det.confidence).substr(0, 4);
        cv::putText(results.finalFrame, text,
                    cv::Point(det.box.x, det.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    // ── Draw speed detections in yellow ───────────────────────────────────
    for (const auto& det : results.speedDetections) {
        cv::rectangle(results.finalFrame, det.box, cv::Scalar(0, 255, 255), 2);
        std::string text = det.label + " " + std::to_string(det.confidence).substr(0, 4);
        cv::putText(results.finalFrame, text,
                    cv::Point(det.box.x, det.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }

    // ── Pedestrian HUD ────────────────────────────────────────────────────
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

    // ── Warning text ──────────────────────────────────────────────────────
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
