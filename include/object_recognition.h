
#include <ros/ros.h>
#include <string>
#include <sensor_msgs/Image.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SURF_train_class.h"

/*****************************************************************************
** Class
*****************************************************************************/
using namespace cv;
using namespace std;

class ObjectRecognition {
public:
	ObjectRecognition(const char *name);
	~ObjectRecognition(void);

	void setFlagShowScreen(bool);
	void setFlagSaveImage(bool);
	const char* getWorkingSpacePath();

    void showImage(Mat image);
    void showDepthImage();
	void showCombineImages();
	void showDepthInRangeImage();
	void showMaxObjectImage();
	// void saveTrainingSet(const char objectName[]);
	void saveObjectImages(const char *objectName);
	Mat getGrayImage();
	Mat getObjectImage();
	void publishMatchObjects(vector<SURFTrainClass::matchData> matchedObjects);

protected:
	FILE *trainFile; 

private:
	ros::Publisher match_publisher;
	ros::Subscriber rgb_image_receiver;
	ros::Subscriber depth_image_receiver;

	Mat rgbImage;
	Mat grayImage;
	Mat depthImage;
	Mat depthInRangeImage;
	Mat dilationImage;
	Mat imageCombine;
	Mat objectImage;
	vector<Rect> boundRect;

	const int minLimitRange;
	const int maxLimitRange;
	bool flagShowScreen;
	bool flagSaveImage;
	char maxContourIndex;
	char imageSavedCount;
	int objectIndex;
	std::string path;
	static const char numTrainingSet = 25;

    void rgbImageCB(const sensor_msgs::ImageConstPtr& msg);
    void depthImageCB(const sensor_msgs::ImageConstPtr& msg);
    void limitRangeDepthImage(int minLimitRange, int maxLimitRange);
    void dilation();
    void objectFusionImage();
    void getObjectContour();
    void drawObjectRectangle(Mat image);
    // void saveObjectImages(Mat image, const char objectName[]);

};

