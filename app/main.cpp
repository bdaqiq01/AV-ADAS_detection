#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "LaneDetect.h"
#include "stopSignDetector.h"
#include "PedestrianDetector.h"

using namespace std;

#define ESCAPE_KEY (27)

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./adas <video_path>" << endl;
        return -1;
    }

    cv::VideoCapture vcCap;
    if (!vcCap.open(argv[1])) {
        cerr << "Failed to open video: " << argv[1] << endl;
        return -1;
    }

    // ── Lane detection (Luke) ────────────────────────────────────────────
    cv::namedWindow("options");
    LaneDetect laneDetect;
    int rho = 1, thetaDivisor = 180, thresholdProb = 50;
    int minLineLength = 50, maxLineGap = 10;
    cv::createTrackbar("Rho",              "options", &rho,           10);
    cv::setTrackbarMin("Rho",              "options", 1);
    cv::createTrackbar("Theta divisor",    "options", &thetaDivisor,  360);
    cv::setTrackbarMin("Theta divisor",    "options", 1);
    cv::createTrackbar("Threshold (prob)", "options", &thresholdProb, 300);
    cv::createTrackbar("Min Line Length",  "options", &minLineLength, 100);
    cv::createTrackbar("Max Line Gap",     "options", &maxLineGap,    100);

    // ── Stop sign detection (Basira) ─────────────────────────────────────
    stopSignDetect stopDetector;
    if (!stopDetector.loadEngine("models/stop.engine")) {
        cerr << "Failed to load stop.engine." << endl;
        return -1;
    }

    // ── Pedestrian detection (Aditya) ────────────────────────────────────
    ped::PedestrianDetector pedDetector;
    if (!pedDetector.loadEngine("models/pedestrian.engine")) {
        cerr << "Failed to load pedestrian.engine." << endl;
        return -1;
    }

    cout << "All modules loaded. Starting pipeline..." << endl;

    while (true) {
        cv::Mat matFrame;
        vcCap.read(matFrame);
        if (matFrame.empty()) {
            cerr << "End of video or empty frame." << endl;
            break;
        }

        // ── Lane detection ───────────────────────────────────────────────
        HoughParams params{ rho, thetaDivisor, thresholdProb, minLineLength, maxLineGap };
        cv::Mat laneFrame = laneDetect.runHough(matFrame, params);

        // ── Stop sign detection ──────────────────────────────────────────
        vector<float> stopRaw = stopDetector.inferRaw(matFrame);

        // ── Pedestrian detection + risk visualization ────────────────────
        auto peds = pedDetector.detect(matFrame);
        pedDetector.visualize(matFrame, peds);

        // ── HUD ──────────────────────────────────────────────────────────
        int n_high = 0, n_med = 0, n_low = 0;
        for (const auto& p : peds) {
            if      (p.risk == ped::RiskLevel::HIGH)   n_high++;
            else if (p.risk == ped::RiskLevel::MEDIUM)  n_med++;
            else                                   n_low++;
        }
        cv::putText(matFrame,
            "Peds: " + to_string(peds.size()) +
            "  H:" + to_string(n_high) +
            " M:" + to_string(n_med) +
            " L:" + to_string(n_low),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        // ── Display ──────────────────────────────────────────────────────
        cv::imshow("ADAS — Pedestrian + Risk", matFrame);
        cv::imshow("ADAS — Lane Detection",    laneFrame);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == ESCAPE_KEY) break;
    }

    cv::destroyAllWindows();
    return 0;
}
