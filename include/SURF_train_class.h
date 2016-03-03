#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
/*****************************************************************************
** Class
*****************************************************************************/
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class SURFTrainClass {
public:
	SURFTrainClass(const char *inputWorkingSpace);
	~SURFTrainClass();
	bool save(const char *szFileTrain);
	Mat getSURFFeatureMat(Mat inputImage);
	void findMatches(Mat inputImage, const char *loadFileName);
	
private:
	const int minHessian;
	const bool bShowMatchImage;

	void writeWorkingSpace(const char *);
	int getPatternNumber(const char *loadFileName);
	bool matchInLimit(KeyPoint objectKeyPoint, KeyPoint imageKeyPoint, int angleRange);
	void showMatchImage(Mat inputImage, std::vector<Point2f> scene_corners);

};


