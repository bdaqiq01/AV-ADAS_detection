#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "LaneDetect.h"
#include "yoloDetector.h"
#include "frameProcessor.h"
#include "PedestrianDetector.h"

using namespace std;

#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)

int main(int argc, char** argv)
{
    cv::VideoCapture vcCap;
    cv::namedWindow("source",  cv::WINDOW_NORMAL);
    cv::namedWindow("final",   cv::WINDOW_NORMAL);

    cv::resizeWindow("source",  640, 360);
    cv::resizeWindow("final",   640, 360);

    cv::moveWindow("source",  360,  20);
    cv::moveWindow("final",   1020, 20);

    int winInput = -1;

    cv::CommandLineParser parser(
        argc, argv,
        "{@input         | | input file}"
        "{camera       c | 0 | camera capture}"
        "{lane-mode    m | 0 | Sliding window (0) or Hough lines (1)}"
        "{dashcam-pos  p | 0 | Adjusts ROI based on per-video dashcam position}"
        "{output-video o | false | Write video}"
        "{debug        d | false | Flag for enabling showing of debug windows}"
        "{disable-lane   | false | Disables lane detection}"
        "{disable-sign   | false | Disables traffic sign detection}"
        "{disable-ped    | false | Disables pedestrian detection}"
    );

    std::string inputFile = parser.get<std::string>("@input");
    int camera = parser.get<int>("camera");
    int ldMode = parser.get<int>("lane-mode");
    int profileIndex = parser.get<int>("dashcam-pos");
    bool enableVideoWrite = parser.get<bool>("output-video");
    bool debugMode = parser.get<bool>("debug");
    bool disableLane = parser.get<bool>("disable-lane");
    bool disableSign = parser.get<bool>("disable-sign");
    bool disablePed = parser.get<bool>("disable-ped");

    // Create frame processor -- disable features if asked
    FrameProcessorOptions fpOpts{disableLane, disableSign, disablePed};
    FrameProcessor frameProcessor(fpOpts);

    // Check whether to run old Hough-based version or new sliding windows version for lane detection
    LaneDetectMode laneMode = (ldMode == 1) ? LaneDetectMode::HoughLines : LaneDetectMode::SlidingWindow;

    // Reads a config file that has values for adjusting the ROI;
    // this is done since some of our videos have different dashcam postions
    // In reality you'd probably have a fixed location and wouldn't need to do this
    cv::FileStorage fs("config/roi_config.yaml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to read ROI config file\n";
        return -1;
    }

    cv::FileNode profile = fs["profiles"][profileIndex];

    ROIConfig laneROI;
    laneROI.yTop = (double)profile["y_top"];
    laneROI.yBottom = (double)profile["y_bottom"];
    laneROI.topLeftX = (double)profile["top_left_x"];
    laneROI.topRightX = (double)profile["top_right_x"];
    laneROI.bottomLeftX = (double)profile["bottom_left_x"];
    laneROI.bottomRightX = (double)profile["bottom_right_x"];

    // Create LaneDetect object with ROI adjustment info
    LaneDetect laneDetect(laneROI);

    if (debugMode) {
        laneDetect.debug = true;
    }

    if (!inputFile.empty()) {
        if (!vcCap.open(inputFile)) {
            vcCap.open(camera);
        }
    } else {
        vcCap.open(camera);
    }

    if (!vcCap.isOpened()) {
        cerr << "Failed to open video source.\n";
        return SYSTEM_ERROR;
    }

    cv::VideoWriter writer;
    
    if (enableVideoWrite) {
        double fps = vcCap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;

        int frameWidth  = static_cast<int>(vcCap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(vcCap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // Use of a pipeline mentioned here:
        // https://forums.developer.nvidia.com/t/opencv-video-write-help-on-nano/183684
        std::string outPipeline =
            "appsrc ! "
            "videoconvert ! "
            "video/x-raw,format=I420 ! "
            "x264enc speed-preset=ultrafast tune=zerolatency bitrate=8000 ! "
            "h264parse ! "
            "mp4mux ! "
            "filesink location=output/final_output.mp4 sync=false";

        writer.open(
            outPipeline, 
            cv::CAP_GSTREAMER, 
            0, 
            fps, 
            cv::Size(frameWidth, frameHeight),
            true
        );

        if (!writer.isOpened()) {
            cerr << "Failed to open output video file.\n";
            return SYSTEM_ERROR;
        }
    }

    vector<string> stopClasses = { "STOP" };
    YoloDetector stopDetector(stopClasses, 0.75f, 0.45f);
    if (!stopDetector.loadEngine("models/stop.engine")) {
        cerr << "Failed to load the stop.engine.\n";
        return -1;
    }

    vector<string> speedClasses = {
        "SPEED 10", "SPEED 15", "SPEED 20", "SPEED 25",
        "SPEED 30", "SPEED 35", "SPEED 40", "SPEED 45",
        "SPEED 50", "SPEED 55", "SPEED 60", "SPEED 65",
        "SPEED 70", "SPEED 75"
    };

    YoloDetector speedDetector(speedClasses, 0.90f, 0.45f);
    if (!disableSign) {
        if (!speedDetector.loadEngine("models/speedlimit.engine")) {
            cerr << "Failed to load the speedlimit.engine.\n";
            return -1;
        }
    }

    ped::PedestrianDetector pedDetector;
    if (!disablePed) {
        if (!pedDetector.loadEngine("models/pedestrian.engine")) {
            cerr << "Failed to load pedestrian.engine.\n";
            return -1;
        }
    }

    cout << "All modules loaded. Starting pipeline..." << endl;

    int frameCount = 0;
    double currentFPS = 0.0;
    auto startTime = chrono::steady_clock::now();
    auto lastFpsPrintTime = startTime;

    while (true) {
        cv::Mat matFrame;
        vcCap.read(matFrame);
        if (matFrame.empty()) {
            cerr << "Error: empty frame received." << endl;
            break;
        }

        frameCount++;

        auto tProcessStart = chrono::steady_clock::now();

        FrameResults results = frameProcessor.processFrame(
            matFrame, 
            laneDetect,
            laneMode, 
            stopDetector, 
            speedDetector, 
            pedDetector);

        auto tProcessEnd = chrono::steady_clock::now();

        auto now = chrono::steady_clock::now();
        double elapsedSeconds =
            chrono::duration_cast<chrono::milliseconds>(now - startTime).count() / 1000.0;
        if (elapsedSeconds > 0.0) currentFPS = frameCount / elapsedSeconds;

        double sinceLastPrint =
            chrono::duration_cast<chrono::milliseconds>(now - lastFpsPrintTime).count() / 1000.0;
        if (sinceLastPrint >= 1.0) {
            double totalMs = chrono::duration<double, milli>(tProcessEnd - tProcessStart).count();
            cout << "Frames: " << frameCount
                 << " | FPS: " << currentFPS
                 << " | process: " << totalMs << "ms" << endl;
            lastFpsPrintTime = now;
        }

        cv::imshow("source", matFrame);
        cv::imshow("final",  results.finalFrame);

        if (enableVideoWrite) writer.write(results.finalFrame);

        winInput = cv::waitKey(1);
        if (winInput == ESCAPE_KEY) {
            break;
        } else if (winInput == 'n') {
            cout << "input " << static_cast<char>(winInput) << " ignored" << endl;
        }
    }

    if (enableVideoWrite) writer.release();
    cv::destroyAllWindows();
    return 0;
}
