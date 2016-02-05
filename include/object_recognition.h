
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
	bool flagSaveImage;
    void showImage(cv::Mat image);
    void showDepthImage();
	void showCombineImages();
	void showDepthInRangeImage();
	void showObjectImage();
	void saveTrainingSet(const char objectName[]);
	std::string path;

protected:
	FILE *trainFile; 

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
	cv::vector<cv::Rect> boundRect;

	char maxContourIndex;
	char imageSavedCount;
	int objectIndex;
	static const char numTrainingSet = 25;

    void rgbImageCB(const sensor_msgs::ImageConstPtr& msg);
    void depthImageCB(const sensor_msgs::ImageConstPtr& msg);
    void limitRangeDepthImage(float minRange, float maxRange);
    void dilation();
    void objectFusionImage();
    void getObjectContour();
    void drawObjectRectangle(cv::Mat image);
    void saveObjectImages(cv::Mat image, const char objectName[]);
};

