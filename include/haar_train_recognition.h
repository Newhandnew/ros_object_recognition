#include <opencv2/highgui/highgui.hpp>

/*****************************************************************************
** Class
*****************************************************************************/
using namespace std;
using namespace cv;


class HaarTrainRecognition {
public:
	HaarTrainRecognition();
	~HaarTrainRecognition();
	bool fSaveEigenImage;
	bool learn(const char *szFileTrain);
	void writeWorkingSpace(const char *);

private:
	// Global variables
	IplImage ** objectImgArr; // array of object images
	vector<string> personNames;			// array of person names (indexed by the person number). Added by Shervin.
	// int objectWidth;	// Default dimensions for objects in the object recognition database. Added by Shervin.
	// int objectHeight;	//	"		"		"		"		"		"
	int nPersons; // the number of people in the training set. Added by Shervin.
	int nTrainObjects; // the number of training images
	int nEigens; // the number of eigenvalues
	IplImage * pAvgTrainImg; // the average image
	IplImage ** eigenVectArr; // eigenvectors
	CvMat * eigenValMat; // eigenvalues
	CvMat * projectedTrainObjectMat; // projected training objects
	// CvHaarClassifierCascade* objectCascade;
	CvMat * trainPersonNumMat;  // the person numbers during training
	bool database_updated;
	//Functions:
	int  loadObjectImgArray(const char * filename);
	IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight);
	void doPCA();
	void storeTrainingData();
	void storeEigenobjectImages();
	
	int  loadTrainingData(CvMat ** pTrainPersonNumMat);
	int  findNearestNeighbor(float * projectedTestObject);
	int  findNearestNeighbor(float * projectedTestObject, float *pConfidence);
	IplImage* convertFloatImageToUcharImage(const IplImage *srcImg);
	// bool retrainOnline(void);
};


