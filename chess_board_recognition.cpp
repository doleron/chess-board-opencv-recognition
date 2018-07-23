#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;
 
int main(int argc, char** argv) {
 
    VideoCapture cap(1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
 
    if (!cap.isOpened()) {
        cout << "Cannot open the video file" << endl;
        return -1;
    }
 
    Mat frame, gray, blur, cannyed;
    VideoWriter output_cap("sample.avi", CV_FOURCC('M', 'J', 'P', 'G'), 12, Size(640, 480));
 
    while (true) {
 
        bool bSuccess = cap.read(frame);
        cap.read(frame);
        if (!bSuccess) {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
 
        cvtColor(frame, gray, CV_BGR2GRAY);
        GaussianBlur(gray, blur, Size(7, 7), 0, 0);
        Canny(blur, cannyed, 50, 200, 3);
        vector<Vec2f> lines;
        HoughLines(cannyed, lines, 1, CV_PI / 180, 120, 0, 0);
 
        Mat blackImage(frame.size(), CV_8UC1, Scalar(0, 0, 0));
        for (size_t i = 0; i < lines.size(); i++) {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            line(blackImage, pt1, pt2, Scalar(255, 255, 255), 1, CV_AA);
        }
 
        vector<vector<Point>> contours;
        findContours(blackImage, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
 
        vector<Moments> mu(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            mu[i] = moments(contours[i], false);
        }
 
        vector<Point2f> mc(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
        }
 
        Mat dst;
         frame.copyTo(dst);
 
        vector<Rect> rects;
        for (int i = 0; i < contours.size(); i++) {
            double a = contourArea(contours[i]);
            if (a > 2000 && a < 3000) {
                drawContours(dst, contours, i, Scalar(255, 0, 0));
                Rect r = boundingRect(contours[i]);
                rects.push_back(r);
            }
        }
        output_cap.write(dst);
        imshow("detected squares", dst);
 
        short int key = waitKey(10);
        if (key == 27) {
             break;
         }
    }
    cap.release();
    output_cap.release();
    return 0;
}
