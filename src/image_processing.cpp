#include <ros/ros.h>
#include <image_processing.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/package.h> //to get pkg path
#include <sys/stat.h> 

ImageProcessing::ImageProcessing() : minLimitRange(30), maxLimitRange(900) {

}

ImageProcessing::ImageProcessing(const char *name) : minLimitRange(30), maxLimitRange(900) {
    imageSavedCount = 0;
    // numTrainingSet(25);
    flagSaveImage = false;
    objectIndex = 1;
    ros::NodeHandle n;
    path = ros::package::getPath(name);     // get pkg path
    rgb_image_receiver = n.subscribe("/camera/rgb/image_raw", 1, &ImageProcessing::rgbImageCB, this);
    depth_image_receiver = n.subscribe("/camera/depth/image_raw", 1, &ImageProcessing::depthImageCB, this);
    if (flagShowScreen) {
        namedWindow( "Show images", CV_WINDOW_AUTOSIZE );
    }
}

ImageProcessing::~ImageProcessing(void) {
    if (flagShowScreen) {
        cvDestroyWindow("Show images");
    }
}

void ImageProcessing::setFlagShowScreen(bool flag) {
    flagShowScreen = flag;
}

void ImageProcessing::setFlagSaveImage(bool flag) {
    flagSaveImage = flag;
}

const char* ImageProcessing::getWorkingSpacePath() {
    return path.c_str();
}

void ImageProcessing::rgbImageCB(const sensor_msgs::ImageConstPtr& pRGBInput) {
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

void ImageProcessing::showImage(Mat image) {
    if(image.cols > 0) {
        imshow("Image", image);
        cvWaitKey(1);
    }
}


void ImageProcessing::depthImageCB(const sensor_msgs::ImageConstPtr& pDepthInput) {
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

void ImageProcessing::showDepthImage() {
    if(depthImage.cols > 0) {
        Mat adjDepthImage;
        double maxRange = 8000;     // set max range to scale down image value
        depthImage.convertTo(adjDepthImage, CV_8UC1, 255 / (maxRange), 0);
        imshow("Depth image", adjDepthImage);
        cvWaitKey(1);
    }
}

void ImageProcessing::showCombineImages() {
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
        if (flagShowScreen) {
            resize(dst, dst, Size(), 0.7, 0.7, INTER_LINEAR);
            imshow("Show images", dst);
        }
        // cvWaitKey(1);
    }
}

void ImageProcessing::limitRangeDepthImage(int minLimitRange, int maxLimitRange) {
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

void ImageProcessing::dilation() {
    if(depthInRangeImage.cols > 0) {
        char dilation_size = 4;        // chnage this setting if necessary
        Mat element = getStructuringElement( MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
        dilate( depthInRangeImage, dilationImage, element );
    }
}

void ImageProcessing::objectFusionImage() {
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

void ImageProcessing::getObjectContour() {
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

void ImageProcessing::drawObjectRectangle(Mat image) {
    if(boundRect.size() > 0) {
        for(int i = 0; i < boundRect.size(); i++) {
            rectangle(image, boundRect[i], Scalar(0, 0, 255), 2, 8, 0);
        }
    }
}

void ImageProcessing::showMaxObjectImage() {
    if (flagShowScreen) {
        if(boundRect.size() > 0) {
            showImage(objectImage(boundRect[maxContourIndex]));
        }
    }
}

Mat ImageProcessing::getGrayImage() {
    return grayImage;
}

Mat ImageProcessing::getObjectImage() {
    return objectImage;
}

// check image exist or not, save image and pattern.txt or rewrite image only
void ImageProcessing::saveObjectImages(const char *objectName) {
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


// ===========================================================================================
