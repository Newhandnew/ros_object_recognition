#include <opencv2/highgui/highgui.hpp>

/*****************************************************************************
** Class
*****************************************************************************/
using namespace std;
using namespace cv;


class HaarTrainRecognition {
public:
	// Global variables
	HaarTrainRecognition();
	~HaarTrainRecognition();
	int SAVE_EIGENFACE_IMAGES;// Set to 0 if you don't want images of the Eigenvectors saved to files (for debugging).
	IplImage ** faceImgArr; // array of face images
	vector<string> personNames;			// array of person names (indexed by the person number). Added by Shervin.
	int faceWidth;	// Default dimensions for faces in the face recognition database. Added by Shervin.
	int faceHeight;	//	"		"		"		"		"		"
	int nPersons; // the number of people in the training set. Added by Shervin.
	int nTrainFaces; // the number of training images
	int nEigens; // the number of eigenvalues
	IplImage * pAvgTrainImg; // the average image
	IplImage ** eigenVectArr; // eigenvectors
	CvMat * eigenValMat; // eigenvalues
	CvMat * projectedTrainFaceMat; // projected training faces
	CvHaarClassifierCascade* faceCascade;
	CvMat * trainPersonNumMat;  // the person numbers during training
	bool database_updated;
	//Functions:
	bool learn(const char *szFileTrain);
	void doPCA();
	void storeTrainingData();
	int  loadTrainingData(CvMat ** pTrainPersonNumMat);
	int  findNearestNeighbor(float * projectedTestFace);
	int  findNearestNeighbor(float * projectedTestFace, float *pConfidence);
	int  loadFaceImgArray(const char * filename);
	void storeEigenfaceImages();
	IplImage* convertImageToGreyscale(const IplImage *imageSrc);
	IplImage* cropImage(const IplImage *img, const CvRect region);
	IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight);
	IplImage* convertFloatImageToUcharImage(const IplImage *srcImg);
	CvRect detectFaceInImage(const IplImage *inputImg, const CvHaarClassifierCascade* cascade );
	bool retrainOnline(void);

	void writeWorkingSpace(char *);

protected:

private:
	char *workingSpacePath;
};


