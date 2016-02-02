
#include <ros/ros.h>
#include <string>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/*****************************************************************************
** Class
*****************************************************************************/

class ObjectRecognition {
public:
	ObjectRecognition(int argc, char** argv);
	~ObjectRecognition(void);
	bool flagShowScreen;
    void showImage(cv::Mat image);
    void showDepthImage();
	void showCombineImages();
	void showDepthInRangeImage();

private:
	ros::Publisher chatter_publisher;
	ros::Publisher face_recognition_command;
	ros::Subscriber face_recognition_feedback;
	ros::Subscriber rgb_image_receiver;
	ros::Subscriber depth_image_receiver;

	cv::Mat rgbImage;
	cv::Mat depthImage;
	cv::Mat depthInRangeImage;
	cv::Mat dilationImage;
	cv::Mat imageCombine;
	cv::Mat objectImage;

    void rgbImageCB(const sensor_msgs::ImageConstPtr& msg);
    void depthImageCB(const sensor_msgs::ImageConstPtr& msg);
    void limitRangeDepthImage(float minRange, float maxRange);
    void dilation();
    void getObjectContour();
};

