/*
 *
 *  Example by Sam Siewert 
 *
 *  Updated for OpenCV 3.1
 *
 */
 #include <unistd.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <iostream>
 
 #include <opencv2/core/core.hpp>
 #include <opencv2/highgui/highgui.hpp>
 #include <opencv2/imgproc/imgproc.hpp>
 
 using namespace cv;
 using namespace std;
 
 #define HRES 640
 #define VRES 480
 #define ESCAPE_KEY (27)
 #define SYSTEM_ERROR (-1)
 
 
 int main(int argc, char** argv)
 {
     char* option; 
     VideoCapture vcap;
     namedWindow("Capture Example");
     Mat frame;
     Mat gray, canny_frame, cdst;
    // vector<Vec4i> lines; //FOR highInes p
     vector<Vec2f> lines; //for lines
     
     if(argc < 2) //parsing the command line i
     {
         cout << "usage: <> camera/video" << endl; 
         return SYSTEM_ERROR; 
     }
     else {
         option = argv[1];
     }
     
     
     if (strcmp(option, "camera") == 0)
     {
         if(!vcap.open(0)) //open the camera device 0 into vcap 
         {
         std::cout << "Error opening video stream or file" << std::endl;
         return -1;
         }
         else
         {
            std::cout << "Opened default camera interface" << std::endl;
         }
     }
     
     else if (strcmp(option, "video") == 0) //open the video into vcap
     
     {
         if(!vcap.open("22400003.mp4")) 
         {
         std::cout << "Error opening video or file" << std::endl;
         return -1;
         }
         else 
         {
             std::cout <<"Opened the video. " << endl;
         }
         
     }
 
 
    if (!vcap.isOpened()) //check if the video or the steam succeded
    {
        exit(SYSTEM_ERROR);
    }
 
    vcap.set(CAP_PROP_FRAME_WIDTH, 640);
    vcap.set(CAP_PROP_FRAME_HEIGHT, 480);
 
     while(1)
     {
         if(!vcap.read(frame) || frame.empty()) break; // end of video or read failure
         Canny(frame, canny_frame, 50, 200, 3);
 
         cvtColor(canny_frame, cdst, COLOR_GRAY2BGR);     
         cvtColor(frame, gray, COLOR_BGR2GRAY);
 
         // Probabilistic Line Transform
         //HoughLinesP(canny_frame, lines, 1, CV_PI/180, 50, 50, 10);
 
             // Standard Hough Line Transform
         HoughLines(canny_frame, lines, 1, CV_PI/180, 200, 0, 0 ); // runs the actual detection
         // Draw the lines
         for( size_t i = 0; i < lines.size(); i++ )
         {
             float rho = lines[i][0], theta = lines[i][1];
             Point pt1, pt2;
             double a = cos(theta), b = sin(theta);
             double x0 = a*rho, y0 = b*rho;
             pt1.x = cvRound(x0 + 1000*(-b));
             pt1.y = cvRound(y0 + 1000*(a));
             pt2.x = cvRound(x0 - 1000*(-b));
             pt2.y = cvRound(y0 - 1000*(a));
             line( cdst, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
         }
 
 
 //        for( size_t i = 0; i < lines.size(); i++ ) //f9r hlinep only 
 //        {
 //          Vec4i l = lines[i];
 //          line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
  //       }
 
 
         //imshow("Capture Example", frame);
 
         imshow("Capture Example", cdst);
 
         char c = waitKey(10);
         if( c == 'q' ) break;
     }
 
     destroyWindow("Capture Example");
     
 };