#include <ros/ros.h>
#include <object_recognition.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/package.h> //to get pkg path
#include <sys/stat.h> 
#include "SURF_train_class.h"

ObjectRecognition::ObjectRecognition(int argc, char** argv) : 
    minLimitRange(30), maxLimitRange(900) {
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle n;
    imageSavedCount = 0;
    // numTrainingSet(25);
    flagSaveImage = false;
    objectIndex = 1;
    path = ros::package::getPath("object_recognition");     // get pkg path
    rgb_image_receiver = n.subscribe("/camera/rgb/image_raw", 1, &ObjectRecognition::rgbImageCB, this);
    depth_image_receiver = n.subscribe("/camera/depth/image_raw", 1, &ObjectRecognition::depthImageCB, this);
    if (flagShowScreen) {
        namedWindow( "Show images", CV_WINDOW_AUTOSIZE );
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
        cvtColor(rgbImage, grayImage, CV_BGR2GRAY);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from ROS images to OpenCV type: %s", e.what());
    }
}

void ObjectRecognition::showImage(Mat image) {
    if(image.cols > 0) {
        imshow("Image", image);
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
        Mat adjDepthImage;
        double maxRange = 8000;     // set max range to scale down image value
        depthImage.convertTo(adjDepthImage, CV_8UC1, 255 / (maxRange), 0);
        imshow("Depth image", adjDepthImage);
        cvWaitKey(1);
    }
}

void ObjectRecognition::showCombineImages() {
    int dstWidth = rgbImage.cols * 3;
    int dstHeight = rgbImage.rows * 2;
    if(dstWidth > 0) {
        // image1: RGB image processing
        Mat dst = Mat(dstHeight, dstWidth, CV_8UC3, Scalar(0,0,0));
        Rect roi(Rect(0, 0, rgbImage.cols, rgbImage.rows));
        Mat targetROI = dst(roi);
        rgbImage.copyTo(targetROI);
        // image2: depth image processing
        Mat adjDepthImage, rgbDepthImage;
        double maxRange = 8000;     // set max range to scale down image value
        depthImage.convertTo(adjDepthImage, CV_8UC1, 255 / (maxRange), 0);
        // change gray image to rgb image
        cvtColor(adjDepthImage, rgbDepthImage, CV_GRAY2RGB);
        targetROI = dst(Rect(rgbImage.cols, 0, depthImage.cols, depthImage.rows));
        rgbDepthImage.copyTo(targetROI);
        // image3: limit range and dilation image processing
        limitRangeDepthImage(minLimitRange, maxLimitRange);
        dilation();
        // change gray image to rgb image and show
        cvtColor(dilationImage, rgbDepthImage, CV_GRAY2RGB);
        targetROI = dst(Rect(depthImage.cols * 2, 0, depthImage.cols, depthImage.rows));
        rgbDepthImage.copyTo(targetROI);
        // image4: image fusion get object contour
        objectFusionImage();
        targetROI = dst(Rect(0, depthImage.rows, depthImage.cols, depthImage.rows));
        objectImage.copyTo(targetROI);  
        // image5: get draw rectangle in rgb image
        getObjectContour();
        drawObjectRectangle(rgbImage);
        targetROI = dst(Rect(rgbImage.cols, rgbImage.rows, rgbImage.cols, rgbImage.rows));
        rgbImage.copyTo(targetROI); 

        // resize
        resize(dst, dst, Size(), 0.7, 0.7, INTER_LINEAR);
        imshow("Show images", dst);
        // cvWaitKey(1);
    }
}

void ObjectRecognition::limitRangeDepthImage(int minLimitRange, int maxLimitRange) {
    if(depthImage.cols > 0) {
        depthInRangeImage = Mat(depthImage.rows, depthImage.cols, CV_8UC1, Scalar::all(0));
        for(int i = 0; i < depthImage.rows; i++) { 
            for(int j = 0; j < depthImage.cols; j++) {
                if(depthImage.at<float>(i,j) > minLimitRange && depthImage.at<float>(i,j) < maxLimitRange) {
                    depthInRangeImage.at<char>(i,j) = 255;
                }
            }   
        }
    }
}

void ObjectRecognition::dilation() {
    if(depthInRangeImage.cols > 0) {
        char dilation_size = 4;        // chnage this setting if necessary
        Mat element = getStructuringElement( MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
        dilate( depthInRangeImage, dilationImage, element );
    }
}

void ObjectRecognition::objectFusionImage() {
    if(dilationImage.cols > 0 && rgbImage.cols >0) {
        // fusion object contour to rgb image
        objectImage = Mat(rgbImage.rows, rgbImage.cols, CV_8UC3, Scalar(0,0,0));
        for(int i = 0; i < dilationImage.rows; i++) { 
            for(int j = 0; j < dilationImage.cols; j++) {
                if(dilationImage.at<char>(i,j) != 0) {
                    objectImage.at<Vec3b>(i,j) = rgbImage.at<Vec3b>(i,j);
                }
            }   
        }
    }
}

void ObjectRecognition::getObjectContour() {
    if(objectImage.cols > 0) {
        // get contour rectangle image
        vector<vector<Point> > contours; // Vector for storing contour
        vector<Vec4i> hierarchy;
        int areaThreshold = 60;
        int maxArea = 0;
        findContours(dilationImage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
        boundRect.resize(contours.size());
        for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
        {
            double area = contourArea(contours[i],false);  //  Find the area of contour
            if(area > areaThreshold){
                boundRect[i] = boundingRect(contours[i]); // Find bounding rectangle
                // get max area index
                if(area > maxArea) {
                    maxArea = area;
                    maxContourIndex = i;
                }
            }
        }
    }
}

void ObjectRecognition::drawObjectRectangle(Mat image) {
    if(boundRect.size() > 0) {
        for(int i = 0; i < boundRect.size(); i++) {
            rectangle(image, boundRect[i], Scalar(0, 0, 255), 2, 8, 0);
        }
    }
}

void ObjectRecognition::showMaxObjectImage() {
    if(boundRect.size() > 0) {
        showImage(objectImage(boundRect[maxContourIndex]));
    }
}

Mat ObjectRecognition::getGrayImage() {
    return grayImage;
}

Mat ObjectRecognition::getObjectImage() {
    return objectImage;
}

// check image exist or not, save image and pattern.txt or rewrite image only
void ObjectRecognition::saveObjectImages(const char *objectName) {
    char saveFileName[256];
    chdir(path.c_str());
    mkdir("data",S_IRWXU | S_IRWXG | S_IRWXO); 
    sprintf(saveFileName, "data/%s.pgm", objectName);
    if( access( saveFileName, F_OK ) == -1 ) {
        // file not exists write pattern.txt file
        trainFile = fopen("pattern.txt", "a+");       // if not exist create it
        fprintf(trainFile, "%s\n", saveFileName);
        fclose(trainFile);
    }
    imwrite(saveFileName, objectImage(boundRect[maxContourIndex]));     // write the largest object
    printf("write image to %s\n", saveFileName);
}

// equalized and save image
// void ObjectRecognition::saveObjectImages(Mat image, const char objectName[]) {
//     char saveFileName[256];
//     chdir(path.c_str());
//     mkdir("data",S_IRWXU | S_IRWXG | S_IRWXO); 
//     sprintf(saveFileName, "data/%d_%s%d.pgm", objectIndex, objectName, imageSavedCount++);
//     // write train.txt file
//     trainFile = fopen("train.txt", "a+");       // if not exist create it
//     fprintf(trainFile, "%d %s %s\n", objectIndex, objectName, saveFileName);
//     fclose(trainFile);
//     // rgb to gray and equalization
//     Mat objectImageProcessed;
//     cvtColor(image, objectImageProcessed, CV_RGB2GRAY);
//     equalizeHist( objectImageProcessed, objectImageProcessed );     // equalization
//     imwrite(saveFileName, objectImageProcessed);
// }

// void ObjectRecognition::saveTrainingSet(const char objectName[]) {
//     if(flagSaveImage == true) {
//         if(imageSavedCount < numTrainingSet) {
//             saveObjectImages(objectImage(boundRect[maxContourIndex]), objectName);
//         }
//         else {
//             flagSaveImage = false;
//             imageSavedCount = 0;
//         }
//     }
// }

// ===========================================================================================

int main(int argc, char** argv)
{
    bool fShowScreen = true;
    float confidence;
    int iNearest;
    Mat featureImage;
    ObjectRecognition object_recognition(argc, argv);
    SURFTrainClass trainModel(object_recognition.getWorkingSpacePath());
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
                    object_recognition.saveObjectImages(argv[1]);
                    break;
                case 'l':   // save
                    // trainModel.save("train.txt");
                    break;
                case 'r':   // recognition
                    trainModel.findMatches(object_recognition.getObjectImage(), "pattern.txt");
                    // objectImage = object_recognition.getObjectImage();
                    // iNearest = trainModel.findNearestNeighbor(&objectImage, &confidence);
                    // printf("result is: %d and confidence: %f\n", iNearest, confidence);
                    break;
                case 'f':
                    featureImage = trainModel.getSURFFeatureMat(object_recognition.getObjectImage());
                    object_recognition.showImage(featureImage);
                    break;
                case 27:    // ESC = 17
                    exit(1);
                    break;
            }
            // object_recognition.saveTrainingSet("object");
        }
        r.sleep();
    }
    return 0;
}
