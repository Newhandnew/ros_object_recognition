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
#include "PCA_train_class.h"

using namespace std;

ObjectRecognition::ObjectRecognition(int argc, char** argv) {
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle n;
    imageSavedCount = 0;
    // numTrainingSet(25);
    flagSaveImage = false;
    objectIndex = 1;
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

void ObjectRecognition::setFlagShowScreen(bool flag) {
    flagShowScreen = flag;
}

void ObjectRecognition::setFlagSaveImage(bool flag) {
    flagSaveImage = flag;
}

const char* ObjectRecognition::getWorkingSpacePath() {
    return path.c_str();
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

void ObjectRecognition::showMaxObjectImage() {
    if(boundRect.size() > 0) {
        showImage(objectImage(boundRect[maxContourIndex]));
    }
}

IplImage ObjectRecognition::getObjectImage() {
    return IplImage(objectImage);
}

// equalized and save image
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

// ============================================================
// void recognition(cv::Mat inputMatImage);
// void recognition(cv::Mat inputMatImage) {
//     IplImage *inputImage = IplImage(inputMatImage);
//     int iNearest, nearest;
//     float confidence;
//     float * projectedTestObject=0;

//     // Check which person it is most likely to be.
//     iNearest = frl.findNearestNeighbor(projectedTestObject, &confidence);
//     nearest  = frl.trainPersonNumMat->data.i[iNearest];
//     //get the desired confidence value from the parameter server
//     ros::param::getCached("~confidence_value", confidence_value);
//     cvFree(&projectedTestObject);
//     if(confidence<confidence_value)
//     {
//         ROS_INFO("Confidence is less than %f was %f, detected object is not considered.",(float)confidence_value, (float)confidence);
//         // add warning message to image
//         text_image.str("");
//         text_image << "Confidence is less than "<< confidence_value;
//         cvPutText(img, text_image.str().c_str(), cvPoint(objectRect.x, objectRect.y + objectRect.height + 25), &font, textColor);
//     }
//     else
//     {
//         // add recognized name to image
//         text_image.str("");
//         text_image <<  frl.personNames[nearest-1].c_str()<<" is recognized";
//         cvPutText(img, text_image.str().c_str(), cvPoint(objectRect.x, objectRect.y + objectRect.height + 25), &font, textColor);
//        //goal is to recognize_once, therefore set as succeeded.
//         if(goal_id_==0)
//         {
//             result_.names.push_back(frl.personNames[nearest-1].c_str());
//             result_.confidence.push_back(confidence);
//             as_.setSucceeded(result_);
//         }
//         //goal is recognize continuous, provide feedback and continue.
//         else
//         {
//             ROS_INFO("detected %s  confidence %f ",  frl.personNames[nearest-1].c_str(),confidence);              
//             feedback_.names.clear();
//             feedback_.confidence.clear();
//             feedback_.names.push_back(frl.personNames[nearest-1].c_str());
//             feedback_.confidence.push_back(confidence);
//             as_.publishFeedback(feedback_);                
//         }                        
//     }
// }

// ===========================================================================================

int main(int argc, char** argv)
{
    bool fShowScreen = true;
    float confidence;
    IplImage objectImage;
    int iNearest;

    ObjectRecognition object_recognition(argc, argv);
    PCATrainClass trainModel(object_recognition.getWorkingSpacePath());
    // trainModel.writeWorkingSpace(object_recognition.getWorkingSpacePath());
    object_recognition.setFlagShowScreen(fShowScreen);
    ros::Rate r(10); // 10 hz
    while(ros::ok()) {
        ros::spinOnce();
        if(fShowScreen) {
            object_recognition.showCombineImages();
            char keyInput = cvWaitKey(1);
            switch(keyInput) {
                case 'o':   // object
                    object_recognition.showMaxObjectImage();
                    break;
                case 's':   // save
                    object_recognition.setFlagSaveImage(true);
                    break;
                case 'l':   // learn
                    trainModel.learn("train.txt");
                    break;
                case 'r':   // recognition
                    objectImage = object_recognition.getObjectImage();
                    iNearest = trainModel.findNearestNeighbor(&objectImage, &confidence);
                    printf("result is: %d and confidence: %f\n", iNearest, confidence);
                    break;
                case 27:    // ESC = 17
                    exit(1);
                    break;
            }
            object_recognition.saveTrainingSet("object");
        }
        r.sleep();
    }
    return 0;
}
