
#include <ros/ros.h>
#include <string>
#include <sensor_msgs/Image.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/*****************************************************************************
** Class
*****************************************************************************/
using namespace cv;
using namespace std;

class ObjectRecognition {
public:
	ObjectRecognition(int argc, char** argv);
	~ObjectRecognition(void);

	void setFlagShowScreen(bool);
	void setFlagSaveImage(bool);
	const char* getWorkingSpacePath();

    void showImage(Mat image);
    void showDepthImage();
	void showCombineImages();
	void showDepthInRangeImage();
	void showMaxObjectImage();
	void saveTrainingSet(const char objectName[]);
	IplImage getObjectImage();
	void featureDetectSURF();

protected:
	FILE *trainFile; 

private:
	ros::Publisher chatter_publisher;
	ros::Publisher face_recognition_command;
	ros::Subscriber rgb_image_receiver;
	ros::Subscriber depth_image_receiver;

	Mat rgbImage;
	Mat depthImage;
	Mat depthInRangeImage;
	Mat dilationImage;
	Mat imageCombine;
	Mat objectImage;
	vector<Rect> boundRect;

	bool flagShowScreen;
	bool flagSaveImage;
	char maxContourIndex;
	char imageSavedCount;
	int objectIndex;
	std::string path;
	static const char numTrainingSet = 25;

    void rgbImageCB(const sensor_msgs::ImageConstPtr& msg);
    void depthImageCB(const sensor_msgs::ImageConstPtr& msg);
    void limitRangeDepthImage(float minRange, float maxRange);
    void dilation();
    void objectFusionImage();
    void getObjectContour();
    void drawObjectRectangle(Mat image);
    void saveObjectImages(Mat image, const char objectName[]);

};

