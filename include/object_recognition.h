
#include <ros/ros.h>
#include <image_processing.h>
#include "object_recognition/match_data.h"
#include "object_recognition/match_data_array.h"
#include "object_recognition/recognition.h"

/*****************************************************************************
** Class
*****************************************************************************/
using namespace cv;
using namespace std;

class ObjectRecognition {
public:
	ObjectRecognition();
	void publishMatchObjects(vector<SURFTrainClass::matchData> matchedObjects);
	void run();
	void close();

protected:


private:
	const bool fShowScreen;
    float confidence;
    Mat featureImage;
    ImageProcessing *image_processing;
    SURFTrainClass *trainModel;
    const char* trainFileName;
    vector<SURFTrainClass::matchData> matchedObjects;

	ros::Publisher match_publisher;
	ros::ServiceServer service_recognition;

	bool recognitionCB(object_recognition::recognition::Request  &req, 
    object_recognition::recognition::Response &res);

};

