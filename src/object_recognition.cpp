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


using namespace std;

ObjectRecognition::ObjectRecognition(int argc, char** argv) {
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle n;
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

void ObjectRecognition::saveObjectImage() {
    
}

void ObjectRecognition::keyInputEvent() {
    char keyInput = cvWaitKey(1);
    switch(keyInput) {
        case 'c':
            showObjectImage();
            break;
        case 's':
            saveObjectImage();
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
        
        }
        r.sleep();
    }
    return 0;
}

//     int calcNumTrainingPerson(const char * filename)
//     {
//         FILE * imgListFile = 0;
//         char imgFilename[512];
//         int iFace, nFaces=0;
//         int person_num=0;
//           // open the input file
//         if( !(imgListFile = fopen(filename, "r")) )
//         {
//         fprintf(stderr, "Can\'t open file %s\n", filename);
//         return 0;
//      }
//           // count the number of faces
//         while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
//         rewind(imgListFile);
//           //count the number of persons
//         for(iFace=0; iFace<nFaces; iFace++)
//         {
//         char personName[256];
//         int personNumber;
//         // read person number (beginning with 1), their name and the image filename.
//         fscanf(imgListFile, "%d %s %s", &personNumber, personName, imgFilename);
//         if (personNumber > person_num) 
//                person_num = personNumber;
//         }
//         fclose(imgListFile); 
//         return (person_num);     
//     }

//     void imageCB(const sensor_msgs::ImageConstPtr& msg)
//     {
//         //to synchronize with executeCB function.
//         //as far as the goal id is 0, 1 or 2, it's active and there is no preempting request, imageCB function is proceed.
//         if (!as_.isActive())      return;
//         if(!mutex_.try_lock()) return;    
//         if(as_.isPreemptRequested())    
//         {
//             ROS_INFO("Goal %d is preempted",goal_id_);
//             as_.setPreempted();
//             mutex_.unlock(); return;
//         }  

//         //get the value of show_screen_flag from the parameter server
//         ros::param::getCached("~show_screen_flag", show_screen_flag); 

//         cv_bridge::CvImagePtr cv_ptr;
//         //convert from ros image format to opencv image format
//         try
//         {
//             cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
//         }
//         catch (cv_bridge::Exception& e)
//         {
//             ROS_ERROR("cv_bridge exception: %s", e.what());
//             as_.setPreempted();
//             ROS_INFO("Goal %d is preempted",goal_id_);
//             mutex_.unlock();
//             return;
//         }
//         ros::Rate r(4);   
//         IplImage img_input = cv_ptr->image;
//         IplImage *img= cvCloneImage(&img_input); 
//         IplImage *greyImg;
//         IplImage *faceImg;
//         IplImage *sizedImg;
//         IplImage *equalizedImg;
//         CvRect faceRect;
//         // Make sure the image is greyscale, since the Eigenfaces is only done on greyscale image.
//         greyImg = frl.convertImageToGreyscale(img);
//         // Perform face detection on the input image, using the given Haar cascade classifier.
//         faceRect = frl.detectFaceInImage(greyImg,frl.faceCascade);
//         // Make sure a valid face was detected.
//         if (faceRect.width < 1) 
//         {
//             ROS_INFO("No face was detected in the last frame"); 
//             if(show_screen_flag)
//             {
//                 cvPutText(img, text_image.str().c_str(), cvPoint(10, faceRect.y + 50), &font, textColor);
//                 cvShowImage("Input", img);
//                 cvWaitKey(1);
//             }
//             sensor_msgs::ImagePtr msgImage = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
//             image_pub.publish(msgImage);        // publish image
//             cvReleaseImage( &greyImg );cvReleaseImage(&img);
//             r.sleep(); 
//             mutex_.unlock(); 
//             return;
//         }
//         // add rectangle to image
//         cvRectangle(img, cvPoint(faceRect.x, faceRect.y), cvPoint(faceRect.x + faceRect.width-1, faceRect.y + faceRect.height-1), CV_RGB(0,255,0), 1, 8, 0);
//         faceImg = frl.cropImage(greyImg, faceRect);  // Get the detected face image.
//         // Make sure the image is the same dimensions as the training images.
//         sizedImg = frl.resizeImage(faceImg, frl.faceWidth, frl.faceHeight);
//         // Give the image a standard brightness and contrast, in case it was too dark or low contrast.
//         equalizedImg = cvCreateImage(cvGetSize(sizedImg), 8, 1); // Create an empty greyscale image
//         cvEqualizeHist(sizedImg, equalizedImg);
//         cvReleaseImage( &greyImg );cvReleaseImage( &faceImg );cvReleaseImage( &sizedImg );  
//         //check again if preempting request is not there!
//         if(as_.isPreemptRequested())    
//         {
//             ROS_INFO("Goal %d is preempted",goal_id_);
//             cvReleaseImage(&equalizedImg);cvReleaseImage(&img);  
//             as_.setPreempted(); 
//             ROS_INFO("Goal %d is preempted",goal_id_);
//             mutex_.unlock(); return;
//         }
//         //goal is add_face_images
//         if( goal_id_== 2)      
//         {
//             if(add_face_count==0)  
//             {
//              //assign the correct number for the new person
//                 person_number = calcNumTrainingPerson("train.txt")+1; 
//             }
//             char cstr[256];
//          sprintf(cstr, "data/%d_%s%d.pgm", person_number, &goal_argument_[0], add_face_count+1);
//             ROS_INFO("Storing the current face of '%s' into image '%s'.", &goal_argument_[0], cstr);
//             //save the new training image of the person
//             cvSaveImage(cstr, equalizedImg, NULL);
//          // Append the new person to the end of the training data.
//          trainFile = fopen("train.txt", "a");
//          fprintf(trainFile, "%d %s %s\n", person_number, &goal_argument_[0], cstr);
//          fclose(trainFile);
//             if(add_face_count==0)  
//             {
//             //get from parameter server how many training imaged should be acquire.
//                 ros::param::getCached("~add_face_number", add_face_number);
//                 if(add_face_number<=0)
//                 {
//                     ROS_INFO("add_face_number parameter is Zero, it is Invalid. One face was added anyway!");
//                     add_face_number=1;
//                 } 
//                 frl.database_updated = false;
//             }
//             // add text to image
//             text_image.str("");
//             text_image <<"A picture of "<< &goal_argument_[0]<< "was added" <<endl;
//             cvPutText(img, text_image.str().c_str(), cvPoint( 10, 50), &font, textColor);
//               //check if enough number of training images for the person has been acquired, then the goal is succeed.
//             if(++add_face_count==add_face_number)    
//          {
                   
//                 result_.names.push_back(goal_argument_); 
//                 as_.setSucceeded(result_);
//                 if (show_screen_flag)
//                 {
//                    cvShowImage("Input", img);
//                    cvWaitKey(1);
//                 }
//                 cvReleaseImage(&equalizedImg); cvReleaseImage(&img);
//                 mutex_.unlock(); return;
//             }   
//             feedback_.names.clear();
//             feedback_.confidence.clear();
//             feedback_.names.push_back(goal_argument_);
//         //      feedback_.confidence.push_back();
//             as_.publishFeedback(feedback_);     
//         }
//          //goal is to recognize person in the video stream
//         if( goal_id_<2 )      
//         {
//             int iNearest, nearest;
//             float confidence;
//             float * projectedTestFace=0;
//             if(!frl.database_updated)
//               //ROS_INFO("Alert: Database is not updated, You better (re)train from images!");       //BEFORE
//              ROS_WARN("Alert: Database is not updated. Please delete \"facedata.xml\" and re-run!"); //AFTER
//             if(frl.nEigens < 1) 
//             {
//                 ROS_INFO("NO database available, goal is Aborted");
//                 cvReleaseImage(&equalizedImg);cvReleaseImage(&img);  
//                 ROS_INFO("Goal %d is Aborted",goal_id_);
//                 as_.setAborted(); 
//                 mutex_.unlock(); return;
//             }
//             // Project the test images onto the PCA subspace
//             projectedTestFace = (float *)cvAlloc( frl.nEigens*sizeof(float) );
//             // project the test image onto the PCA subspace
//             cvEigenDecomposite(equalizedImg,frl.nEigens,frl.eigenVectArr,0, 0,frl.pAvgTrainImg,projectedTestFace);
//             // Check which person it is most likely to be.
//             iNearest = frl.findNearestNeighbor(projectedTestFace, &confidence);
//             nearest  = frl.trainPersonNumMat->data.i[iNearest];
//             //get the desired confidence value from the parameter server
//             ros::param::getCached("~confidence_value", confidence_value);
//             cvFree(&projectedTestFace);
//             if(confidence<confidence_value)
//             {
//                 ROS_INFO("Confidence is less than %f was %f, detected face is not considered.",(float)confidence_value, (float)confidence);
//                 // add warning message to image
//                 text_image.str("");
//                 text_image << "Confidence is less than "<< confidence_value;
//                 cvPutText(img, text_image.str().c_str(), cvPoint(faceRect.x, faceRect.y + faceRect.height + 25), &font, textColor);
//             }
//             else
//             {
//                 // add recognized name to image
//                 text_image.str("");
//                 text_image <<  frl.personNames[nearest-1].c_str()<<" is recognized";
//                 cvPutText(img, text_image.str().c_str(), cvPoint(faceRect.x, faceRect.y + faceRect.height + 25), &font, textColor);
//             //goal is to recognize_once, therefore set as succeeded.
//                 if(goal_id_==0)
//                 {
//                     result_.names.push_back(frl.personNames[nearest-1].c_str());
//                     result_.confidence.push_back(confidence);
//                     as_.setSucceeded(result_);
//                 }
//                 //goal is recognize continuous, provide feedback and continue.
//                 else
//                 {
//                  ROS_INFO("detected %s  confidence %f ",  frl.personNames[nearest-1].c_str(),confidence);              
//                     feedback_.names.clear();
//                     feedback_.confidence.clear();
//                     feedback_.names.push_back(frl.personNames[nearest-1].c_str());
//                     feedback_.confidence.push_back(confidence);
//                     as_.publishFeedback(feedback_);                
//                 }                        
//             }
//         }
//     }