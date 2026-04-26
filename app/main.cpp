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
#include "stopSignDetector.h"

#include "stopSignDetector.h" //class for the stop sing detection 
using namespace std; 

#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)

int main(int argc, char** argv)
{
    cv::VideoCapture vcCap;
    cv::namedWindow("options",cv::WINDOW_NORMAL );
    cv::namedWindow("source", cv::WINDOW_NORMAL);
    cv::namedWindow("final", cv::WINDOW_NORMAL);

    cv::resizeWindow("options", 320, 300);
    cv::resizeWindow("source", 640, 360);
    cv::resizeWindow("final", 640, 360);
    
    cv::moveWindow("options", 20, 20);
    cv::moveWindow("source", 360, 20);
    cv::moveWindow("final", 1020, 20);

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

	//opening camera of video input 
   if (argc > 1) {
            if (!vcCap.open(argv[1])) {
                vcCap.open(camera);
            }
        } else {
            vcCap.open(camera);
        }
    
        if (!vcCap.isOpened()) {
            cerr << "Failed to open video source.\n";
            return SYSTEM_ERROR;
        }
    


	//writing camera of video ouput
	bool enableVideoWrite = false;
	double fps = vcCap.get(cv::CAP_PROP_FPS);
	if (fps <= 0) fps = 30.0;
	
	int frameWidth = static_cast<int>(vcCap.get(cv::CAP_PROP_FRAME_WIDTH));
	int frameHeight = static_cast<int>(vcCap.get(cv::CAP_PROP_FRAME_HEIGHT));

	cv::VideoWriter writer;
	 if (enableVideoWrite)
	 { 
	 	writer.open("output/final_output.mp4",
	 	            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
	 	            fps,
	 	            cv::Size(frameWidth, frameHeight));
	 	
	 	if (!writer.isOpened()) 
	 		{
	 	    cerr << "Failed to open output video file.\n";
	 	    return -1;
	 		}	
	 }
	
    
    //loading the stop sing engine
    stopSignDetect stopDetector; 
    if (!stopDetector.loadEngine("models/stop.engine")) //checking for the engine fail
    {
        cerr << "Failed to load the stop.engine. \n";
        return -1;
    }
 


    //FPS tracking 
    int frameCount = 0;
    double currentFPS = 0.0;

    auto startTime = chrono::steady_clock::now(); //starter time 
    auto lastFpsPrintTime = startTime;

    while (true) {
        cv::Mat matFrame;
        vcCap.read(matFrame);

        if (matFrame.empty()) {
            std::cerr << "Error: empty frame received." << std::endl;
            break;
        }

		frameCount ++;
		
        HoughParams params{
            rho,
            thetaDivisor,
            thresholdProb,
            minLineLength,
            maxLineGap
        };

		//caling the lane detector 
        cv::Mat finalFrame = laneDetect.runHough(matFrame, params);
        
		//stop sign detector 
        vector<Detection> detections = stopDetector.detect(matFrame);
        
        //drawing the bb on the final fram 
        for (const auto& det : detections) {
            cv::rectangle(finalFrame, det.box, cv::Scalar(0, 255, 0), 2);
        
            string text = det.label + " " + to_string(det.confidence).substr(0, 4);
            cv::putText(finalFrame,
                        text,
                        cv::Point(det.box.x, det.box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(0, 255, 0),
                        2);
        }
	
		//FPS calculation
		auto now = chrono::steady_clock::now();
        double elapsedSeconds =
            chrono::duration_cast<chrono::milliseconds>(now - startTime).count() / 1000.0;

        if (elapsedSeconds > 0.0) {
            currentFPS = frameCount / elapsedSeconds;
        }

        // Print FPS once per second
        double sinceLastPrint =
            chrono::duration_cast<chrono::milliseconds>(now - lastFpsPrintTime).count() / 1000.0;

        if (sinceLastPrint >= 1.0) {
            cout << "Frames processed: " << frameCount
                 << " | Average FPS: " << currentFPS << endl;
            lastFpsPrintTime = now;
        } 



        cv::imshow("source", matFrame);
        cv::imshow("final", finalFrame);

        //writing the frame to the video 
       if (enableVideoWrite)
       {
       		writer.write(finalFrame);
       }
     

        winInput = static_cast<char>(cv::waitKey(1));
        if (winInput == ESCAPE_KEY) {
            break;
        } else if (winInput == 'n') {
            std::cout << "input " << winInput << " ignored" << std::endl;
        }

        
    }

    if (enableVideoWrite) {
     	writer.release(); //close the writer 
     }

    cv::destroyAllWindows();
    return 0;
}
