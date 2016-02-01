
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
    void showRGBImage();
    void showDepthImage();
	void showCombineImages();

private:
	ros::Publisher chatter_publisher;
	ros::Publisher face_recognition_command;
	ros::Subscriber face_recognition_feedback;
	ros::Subscriber rgb_image_receiver;
	ros::Subscriber depth_image_receiver;

	cv::Mat rgbImage;
	cv::Mat depthImage;
	cv::Mat imageCombine;

    void rgbImageCB(const sensor_msgs::ImageConstPtr& msg);
    void depthImageCB(const sensor_msgs::ImageConstPtr& msg);


};

