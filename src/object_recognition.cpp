#include <ros/ros.h>
#include <object_recognition.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
// #include <vector>
// #include <cvaux.h>
// #include <cxcore.hpp>
// #include <sys/stat.h>
// #include <termios.h>
//#include <term.h>
#include <ros/package.h> //to get pkg path
#include <sys/stat.h> 

using namespace std;

ObjectRecognition::ObjectRecognition(int argc, char** argv) {
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle n;
    imageSavedCount = 0;
    // numTrainingSet(25);
    flagSaveImage = false;
    objectIndex = 0;
    path = ros::package::getPath("object_recognition");     // get pkg path
    // face_recognition_feedback = n.subscribe("/face_recognition/feedback", 10, &QNode::feedbackCB, this);
    rgb_image_receiver = n.subscribe("/camera/rgb/image_raw", 1, &ObjectRecognition::rgbImageCB, this);
    depth_image_receiver = n.subscribe("/camera/depth/image_raw", 1, &ObjectRecognition::depthImageCB, this);
 
    if (flagShowScreen) {
        cvNamedWindow( "Show images", CV_WINDOW_AUTOSIZE );
    }
}

ObjectRecognition::~ObjectRecognition(void) {
    if (flagShowScreen) {
        cvDestroyWindow("Show images");
    }
}

void ObjectRecognition::rgbImageCB(const sensor_msgs::ImageConstPtr& pRGBInput) {
    // Convert ROS images to OpenCV
    try
    {
        rgbImage = cv_bridge::toCvShare(pRGBInput, "bgr8") -> image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from ROS images to OpenCV type: %s", e.what());
    }
}

void ObjectRecognition::showImage(cv::Mat image) {
    if(image.cols > 0) {
        cv::imshow("Image", image);
        cvWaitKey(1);
    }
}


void ObjectRecognition::depthImageCB(const sensor_msgs::ImageConstPtr& pDepthInput) {
    // Convert ROS images to OpenCV
    try
    {
        depthImage = cv_bridge::toCvShare(pDepthInput, "32FC1") -> image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from ROS images to OpenCV type: %s", e.what());
    }
}

void ObjectRecognition::showDepthImage() {
    if(depthImage.cols > 0) {
        cv::Mat adjDepthImage;
        double maxRange = 8000;     // set max range to scale down image value
        depthImage.convertTo(adjDepthImage, CV_8UC1, 255 / (maxRange), 0);
        cv::imshow("Depth image", adjDepthImage);
        cvWaitKey(1);
    }
}

void ObjectRecognition::showCombineImages() {
    int dstWidth = rgbImage.cols * 3;
    int dstHeight = rgbImage.rows * 2;
    if(dstWidth > 0) {
        // image1: RGB image processing
        cv::Mat dst = cv::Mat(dstHeight, dstWidth, CV_8UC3, cv::Scalar(0,0,0));
        cv::Rect roi(cv::Rect(0, 0, rgbImage.cols, rgbImage.rows));
        cv::Mat targetROI = dst(roi);
        rgbImage.copyTo(targetROI);
        // image2: depth image processing
        cv::Mat adjDepthImage, rgbDepthImage;
        double maxRange = 8000;     // set max range to scale down image value
        depthImage.convertTo(adjDepthImage, CV_8UC1, 255 / (maxRange), 0);
        // change gray image to rgb image
        cv::cvtColor(adjDepthImage, rgbDepthImage, CV_GRAY2RGB);
        targetROI = dst(cv::Rect(rgbImage.cols, 0, depthImage.cols, depthImage.rows));
        rgbDepthImage.copyTo(targetROI);
        // image3: limit range and dilation image processing
        limitRangeDepthImage(200, 1000);
        dilation();
        // change gray image to rgb image and show
        cv::cvtColor(dilationImage, rgbDepthImage, CV_GRAY2RGB);
        targetROI = dst(cv::Rect(depthImage.cols * 2, 0, depthImage.cols, depthImage.rows));
        rgbDepthImage.copyTo(targetROI);
        // image4: image fusion get object contour
        objectFusionImage();
        targetROI = dst(cv::Rect(0, depthImage.rows, depthImage.cols, depthImage.rows));
        objectImage.copyTo(targetROI);  
        // image5: get draw rectangle in rgb image
        getObjectContour();
        drawObjectRectangle(rgbImage);
        targetROI = dst(cv::Rect(rgbImage.cols, rgbImage.rows, rgbImage.cols, rgbImage.rows));
        rgbImage.copyTo(targetROI); 

        // resize
        cv::resize(dst, dst, cv::Size(), 0.7, 0.7, cv::INTER_LINEAR);
        cv::imshow("Show images", dst);
        // cvWaitKey(1);
    }
}

void ObjectRecognition::limitRangeDepthImage(float minRange, float maxRange) {
    if(depthImage.cols > 0) {
        depthInRangeImage = cv::Mat(depthImage.rows, depthImage.cols, CV_8UC1, cv::Scalar::all(0));
        for(int i = 0; i < depthImage.rows; i++) { 
            for(int j = 0; j < depthImage.cols; j++) {
                if(depthImage.at<float>(i,j) > minRange && depthImage.at<float>(i,j) < maxRange) {
                    depthInRangeImage.at<char>(i,j) = 255;
                }
            }   
        }
    }
}

void ObjectRecognition::dilation() {
    if(depthInRangeImage.cols > 0) {
        char dilation_size = 3;        // chnage this setting if necessary
        cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
        cv::dilate( depthInRangeImage, dilationImage, element );
    }
}

void ObjectRecognition::objectFusionImage() {
    if(dilationImage.cols > 0 && rgbImage.cols >0) {
        // fusion object contour to rgb image
        objectImage = cv::Mat(rgbImage.rows, rgbImage.cols, CV_8UC3, cv::Scalar(0,0,0));
        for(int i = 0; i < dilationImage.rows; i++) { 
            for(int j = 0; j < dilationImage.cols; j++) {
                if(dilationImage.at<char>(i,j) != 0) {
                    objectImage.at<cv::Vec3b>(i,j) = rgbImage.at<cv::Vec3b>(i,j);
                }
            }   
        }
    }
}

void ObjectRecognition::getObjectContour() {
    if(objectImage.cols > 0) {
        // get contour rectangle image
        cv::vector<cv::vector<cv::Point> > contours; // Vector for storing contour
        cv::vector<cv::Vec4i> hierarchy;
        int areaThreshold = 60;
        int maxArea = 0;
        cv::findContours(dilationImage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
        boundRect.resize(contours.size());
        for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
        {
            double area = cv::contourArea(contours[i],false);  //  Find the area of contour
            if(area > areaThreshold){
                boundRect[i] = cv::boundingRect(contours[i]); // Find bounding rectangle
                // get max area index
                if(area > maxArea) {
                    maxArea = area;
                    maxContourIndex = i;
                }
            }
        }
    }
}

void ObjectRecognition::drawObjectRectangle(cv::Mat image) {
    if(boundRect.size() > 0) {
        for(int i = 0; i < boundRect.size(); i++) {
            rectangle(image, boundRect[i], cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }
}

void ObjectRecognition::showObjectImage() {
    if(boundRect.size() > 0) {
        showImage(objectImage(boundRect[maxContourIndex]));
    }
}

void ObjectRecognition::saveObjectImages(cv::Mat image, const char objectName[]) {
    char saveFileName[256];
    chdir(path.c_str());
    mkdir("data",S_IRWXU | S_IRWXG | S_IRWXO); 
    sprintf(saveFileName, "data/%d_%s%d.pgm", objectIndex, objectName, imageSavedCount++);
    // write train.txt file
    trainFile = fopen("train.txt", "a+");       // if not exist create it
    fprintf(trainFile, "%d %s %s\n", objectIndex, objectName, saveFileName);
    fclose(trainFile);
    // rgb to gray and equalization
    cv::Mat objectImageProcessed;
    cv::cvtColor(image, objectImageProcessed, CV_RGB2GRAY);
    cv::equalizeHist( objectImageProcessed, objectImageProcessed );     // equalization
    cv::imwrite(saveFileName, objectImageProcessed);
}

void ObjectRecognition::saveTrainingSet(const char objectName[]) {
    if(flagSaveImage == true) {
        if(imageSavedCount < numTrainingSet) {
            saveObjectImages(objectImage(boundRect[maxContourIndex]), objectName);
        }
        else {
            flagSaveImage = false;
            imageSavedCount = 0;
        }
    }
}

void ObjectRecognition::keyInputEvent() {
    char keyInput = cvWaitKey(1);
    switch(keyInput) {
        case 'c':
            showObjectImage();
            break;
        case 's':
            flagSaveImage = true;
            break;
        case 'l':
            printf("dir: %s", path.c_str());
            break;
        case 27:    // ESC = 17
            exit(1);
            break;
    }
}

// ===========================================================================================

int main(int argc, char** argv)
{
    ObjectRecognition object_recognition(argc, argv);
    object_recognition.flagShowScreen = 1;
    ros::Rate r(10); // 10 hz
    while(ros::ok()) {
        ros::spinOnce();
        if(object_recognition.flagShowScreen) {
            object_recognition.showCombineImages();
            object_recognition.keyInputEvent();
            object_recognition.saveTrainingSet("object");
        }
        r.sleep();
    }
    return 0;
}
