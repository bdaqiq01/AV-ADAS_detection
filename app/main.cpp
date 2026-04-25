#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "LaneDetect.h"
#include "stopSignDetector.h"

#include "stopSignDetector.h" //class for the stop sing detection 
using namespace std; 

// See www.asciitable.com
#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)

int main(int argc, char** argv)
{
    cv::VideoCapture vcCap;
    cv::namedWindow("options");
    cv::namedWindow("source");
    cv::namedWindow("hough probabilistic");

    char winInput;
    LaneDetect laneDetect;

    cv::CommandLineParser parser(
        argc, argv,
        "{@input   | | input file}"
        "{camera c | 0 | camera capture}"
    );

    int camera = parser.get<int>("camera");

    int rho = 1;
    int thetaDivisor = 180;
    int thresholdProb = 50;
    int minLineLength = 50;
    int maxLineGap = 10;

    cv::createTrackbar("Rho", "options", &rho, 10);
    cv::setTrackbarMin("Rho", "options", 1);

    cv::createTrackbar("Theta divisor", "options", &thetaDivisor, 360);
    cv::setTrackbarMin("Theta divisor", "options", 1);

    cv::createTrackbar("Threshold (probabilistic)", "options", &thresholdProb, 300);
    cv::createTrackbar("Min Line Length (probabilistic)", "options", &minLineLength, 100);
    cv::createTrackbar("Max Line Gap (probabilistic)", "options", &maxLineGap, 100);

    if (!vcCap.open(argv[1])) {
        vcCap.open(camera);
    }

    //loading the stop sing engine
    stopSignDetect stopDetector; 
    if (!stopDetector.loadEngine("models/stop.engine")) //checking for the engine fail
    {
        cerr << "Failed to load the stop.engine. \n";
        return -1;
    }
    int frameCount = 0;

    while (true) {
        cv::Mat matFrame;
        vcCap.read(matFrame);

        if (matFrame.empty()) {
            std::cerr << "Error: empty frame received." << std::endl;
            break;
        }

        HoughParams params{
            rho,
            thetaDivisor,
            thresholdProb,
            minLineLength,
            maxLineGap
        };

        cv::Mat matHoughFrame = laneDetect.runHough(matFrame, params);

        vector<Detection> detections = stopDetector.detect(matFrame);
        
        for (const auto& det : detections) {
            cv::rectangle(matFrame, det.box, cv::Scalar(0, 255, 0), 2);
        
            string text = det.label + " " + to_string(det.confidence).substr(0, 4);
            cv::putText(matFrame,
                        text,
                        cv::Point(det.box.x, det.box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(0, 255, 0),
                        2);
        }

        //stop sing 
        vector <float> rawoutput = stopDetector.inferRaw(matFrame);
        if (frameCount % 30 == 0) {
            cout << "Raw output size: " << rawoutput.size() << endl;
        }
        frameCount +=1;

        
        if (!rawoutput.empty()) {
            cout << "First 10 values: ";
            size_t limit = std::min<size_t>(10, rawoutput.size());
            for (size_t i = 0; i < limit; i++) {
                cout << rawoutput[i] << " ";
            }
            cout << endl;
        }
        


        cv::imshow("source", matFrame);
        cv::imshow("hough probabilistic", matHoughFrame);

        winInput = static_cast<char>(cv::waitKey(1));
        if (winInput == ESCAPE_KEY) {
            break;
        } else if (winInput == 'n') {
            std::cout << "input " << winInput << " ignored" << std::endl;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
