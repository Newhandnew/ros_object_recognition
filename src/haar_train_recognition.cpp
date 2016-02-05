#include <cvaux.h>
#include "haar_train_recognition.h"
#include <unistd.h>

//#define USE_MAHALANOBIS_DISTANCE	// You might get better recognition accuracy if you enable this.

HaarTrainRecognition::HaarTrainRecognition() {
	fSaveEigenImage = 1;		// Set to 0 if you don't want images of the Eigenvectors saved to files (for debugging).
	objectImgArr = 0;
	// objectWidth = 120;
	// objectHeight = 90;
	nPersons = 0;
	nTrainObjects = 0;
	nEigens = 0;
	pAvgTrainImg = 0;
	eigenVectArr = 0;
	eigenValMat = 0;
	projectedTrainObjectMat = 0;
	database_updated = false;

	// Load the previously saved training data
	trainPersonNumMat = 0;
	// if ( loadTrainingData( &trainPersonNumMat ) )
	// {
	// 	// objectWidth = pAvgTrainImg->width;
	// 	// objectHeight = pAvgTrainImg->height;
	// 	database_updated = true;
	// }
	// else
	// {
	// 	printf("Will try to train from images");
	// 	if (!retrainOnline())
	// 		printf("Could not train from images");

	// }
}
HaarTrainRecognition::~HaarTrainRecognition(void) {
	// cvReleaseHaarClassifierCascade( &objectCascade );
	if (trainPersonNumMat)	cvReleaseMat( &trainPersonNumMat );
	int i;
	if (objectImgArr)
	{
		for (i = 0; i < nTrainObjects; i++)
			if (objectImgArr[i])	cvReleaseImage( &objectImgArr[i] );
		cvFree( &objectImgArr );
	}
	if (eigenVectArr)
	{
		for (i = 0; i < nEigens; i++)
			if (eigenVectArr[i])      cvReleaseImage( &eigenVectArr[i] );
		cvFree( &eigenVectArr );
	}
	if (trainPersonNumMat) cvReleaseMat( &trainPersonNumMat );
	personNames.clear();
	if (pAvgTrainImg) cvReleaseImage( &pAvgTrainImg );
	if (eigenValMat)  cvReleaseMat( &eigenValMat );
	if (projectedTrainObjectMat) cvReleaseMat( &projectedTrainObjectMat );
}

void HaarTrainRecognition::writeWorkingSpace(const char *inputWorkingSpace) {
	printf("Current working space path is: %s\n", inputWorkingSpace);
	chdir(inputWorkingSpace);
}


// Creates a new image copy that is of a desired size.
// Remember to free the new image later.
IplImage* HaarTrainRecognition::resizeImage(const IplImage *origImg, int newWidth, int newHeight)
{
	IplImage *outImg = 0;
	int origWidth;
	int origHeight;
	if (origImg) {
		origWidth = origImg->width;
		origHeight = origImg->height;
	}
	if (newWidth <= 0 || newHeight <= 0 || origImg == 0 || origWidth <= 0 || origHeight <= 0) {
		printf("ERROR in resizeImage: Bad desired image size of %dx%d.", newWidth, newHeight);
		exit(1);
	}

	// Scale the image to the new dimensions, even if the aspect ratio will be changed.
	outImg = cvCreateImage(cvSize(newWidth, newHeight), origImg->depth, origImg->nChannels);
	if (newWidth > origImg->width && newHeight > origImg->height) {
		// Make the image larger
		cvResetImageROI((IplImage*)origImg);
		cvResize(origImg, outImg, CV_INTER_LINEAR);	// CV_INTER_CUBIC or CV_INTER_LINEAR is good for enlarging
	}
	else {
		// Make the image smaller
		cvResetImageROI((IplImage*)origImg);
		cvResize(origImg, outImg, CV_INTER_AREA);	// CV_INTER_AREA is good for shrinking / decimation, but bad at enlarging.
	}

	return outImg;
}

// Get an 8-bit equivalent of the 32-bit Float image.
// Returns a new image, so remember to call 'cvReleaseImage()' on the result.
IplImage* HaarTrainRecognition::convertFloatImageToUcharImage(const IplImage *srcImg)
{
	IplImage *dstImg = 0;
	if ((srcImg) && (srcImg->width > 0 && srcImg->height > 0)) {

		// Spread the 32bit floating point pixels to fit within 8bit pixel range.
		double minVal, maxVal;
		cvMinMaxLoc(srcImg, &minVal, &maxVal);

		//cout << "FloatImage:(minV=" << minVal << ", maxV=" << maxVal << ")." << endl;

		// Deal with NaN and extreme values, since the DFT seems to give some NaN results.
		if (cvIsNaN(minVal) || minVal < -1e30)
			minVal = -1e30;
		if (cvIsNaN(maxVal) || maxVal > 1e30)
			maxVal = 1e30;
		if (maxVal - minVal == 0.0f)
			maxVal = minVal + 0.001;	// remove potential divide by zero errors.

		// Convert the format
		dstImg = cvCreateImage(cvSize(srcImg->width, srcImg->height), 8, 1);
		cvConvertScale(srcImg, dstImg, 255.0 / (maxVal - minVal), - minVal * 255.0 / (maxVal - minVal));
	}
	return dstImg;
}

// Train from the data in the given text file, and store the trained data into the file 'objectdata.xml'.
bool HaarTrainRecognition::learn(const char *szFileTrain)
{
	int i, offset;

	// load training data
	printf("Loading the training images in '%s'", szFileTrain);
	nTrainObjects = loadObjectImgArray(szFileTrain);
	printf("Got %d training images.\n", nTrainObjects);
	if ( nTrainObjects < 2 )
	{
		fprintf(stderr,
		        "Need 2 or more training objects"
		        "Input file contains only %d", nTrainObjects);
		return false;
	}

	// do PCA on the training objects
	doPCA();

	// project the training images onto the PCA subspace
	projectedTrainObjectMat = cvCreateMat( nTrainObjects, nEigens, CV_32FC1 );
	offset = projectedTrainObjectMat->step / sizeof(float);
	for (i = 0; i < nTrainObjects; i++)
	{
		//int offset = i * nEigens;
		cvEigenDecomposite(
		    objectImgArr[i],
		    nEigens,
		    eigenVectArr,
		    0, 0,
		    pAvgTrainImg,
		    //projectedTrainObjectMat->data.fl + i*nEigens);
		    projectedTrainObjectMat->data.fl + i * offset);
	}

	// store the recognition data as an xml file
	storeTrainingData();

	// Save all the eigenvectors as images, so that they can be checked.
	if (fSaveEigenImage) {
		storeEigenobjectImages();
	}
	return true;

}

// Read the names & image filenames of people from a text file, and load all those images listed.
int HaarTrainRecognition::loadObjectImgArray(const char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iObject, nObjects = 0;
	int i;
	IplImage *pobjectImg;
	IplImage *psizedImg;
	IplImage *pequalizedImg;
	int imgWidth;
	int imgHeight;
	// open the input file
	if ( !(imgListFile = fopen(filename, "r")) )
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// count the number of objects
	while ( fgets(imgFilename, 512, imgListFile) ) ++nObjects;
	rewind(imgListFile);

	// allocate the object-image array and person number matrix
	objectImgArr        = (IplImage **)cvAlloc( nObjects * sizeof(IplImage *) );
	trainPersonNumMat = cvCreateMat( 1, nObjects, CV_32SC1 );

	personNames.clear();	// Make sure it starts as empty.
	nPersons = 0;

	// store the object images in an array
	for (iObject = 0; iObject < nObjects; iObject++)
	{
		char personName[256];
		string sPersonName;
		int personNumber;
		// read person number (beginning with 1), their name and the image filename.
		fscanf(imgListFile, "%d %s %s", &personNumber, personName, imgFilename);
		sPersonName = personName;
		//printf("Got %d: %d, <%s>, <%s>.\n", iObject, personNumber, personName, imgFilename);

		// Check if a new person is being loaded.
		if (personNumber > nPersons) {
			// Allocate memory for the extra person (or possibly multiple), using this new person's name.
			for (i = nPersons; i < personNumber; i++) {
				personNames.push_back( sPersonName );
			}
			nPersons = personNumber;
			//printf("Got new person <%s> -> nPersons = %d [%d]\n", sPersonName.c_str(), nPersons, personNames.size());
		}

		// Keep the data
		trainPersonNumMat->data.i[iObject] = personNumber;

		// load the object image
		pobjectImg = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);
		if(iObject == 0) {
			imgWidth = pobjectImg->width;
			imgHeight = pobjectImg->height;
		}
		psizedImg = resizeImage(pobjectImg, imgWidth, imgHeight);
		// Give the image a standard brightness and contrast, in case it was too dark or low contrast.
		pequalizedImg = cvCreateImage(cvGetSize(psizedImg), 8, 1);	// Create an empty greyscale image
		cvEqualizeHist(psizedImg, pequalizedImg);
		objectImgArr[iObject] = pequalizedImg;
		cvReleaseImage( &psizedImg ); cvReleaseImage( &pobjectImg );
		if ( !objectImgArr[iObject] )
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
	}

	fclose(imgListFile);

	printf("Data loaded from '%s': (%d images of %d people).\n", filename, nObjects, nPersons);
	printf("object: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i = 1; i < nPersons; i++) {
		printf(", <%s>", personNames[i].c_str());
	}
	printf(".\n");

	return nObjects;
}


// Do the Principal Component Analysis, finding the average image
// and the eigenobjects that represent any image in the given dataset.
void HaarTrainRecognition::doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize objectImgSize;

	// set the number of eigenvalues to use
	nEigens = nTrainObjects - 1;

	// allocate the eigenvector images
	objectImgSize.width  = objectImgArr[0]->width;
	objectImgSize.height = objectImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for (i = 0; i < nEigens; i++)
		eigenVectArr[i] = cvCreateImage(objectImgSize, IPL_DEPTH_32F, 1);

	// allocate the eigenvalue array
	eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// allocate the averaged image
	pAvgTrainImg = cvCreateImage(objectImgSize, IPL_DEPTH_32F, 1);

	// set the PCA termination criterion
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);
	printf("**** nTrainObjects: %d", nTrainObjects);
	// compute average image, eigenvalues, and eigenvectors
	cvCalcEigenObjects(
	    nTrainObjects,
	    (void*)objectImgArr,
	    (void*)eigenVectArr,
	    CV_EIGOBJ_NO_CALLBACK,
	    0,
	    0,
	    &calcLimit,
	    pAvgTrainImg,
	    eigenValMat->data.fl);

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

// Save the training data to the file 'objectdata.xml'.
void HaarTrainRecognition::storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interobject
	fileStorage = cvOpenFileStorage( "objectdata.xml", 0, CV_STORAGE_WRITE );

	// Store the person names. Added by Shervin.
	cvWriteInt( fileStorage, "nPersons", nPersons );
	for (i = 0; i < nPersons; i++) {
		char varname[200];
		sprintf( varname, "personName_%d", (i + 1) );
		cvWriteString(fileStorage, varname, personNames[i].c_str(), 0);
	}

	// store all the data
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainObjects", nTrainObjects );
	cvWrite(fileStorage, "trainPersonNumMat", trainPersonNumMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "projectedTrainObjectMat", projectedTrainObjectMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0, 0));
	for (i = 0; i < nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0, 0));
	}

	// release the file-storage interobject
	cvReleaseFileStorage( &fileStorage );
}

void HaarTrainRecognition::storeEigenobjectImages()
{
	// Store the average image to a file
	printf("Saving the image of the average object as 'out_averageImage.bmp'.");
	cvSaveImage("out_averageImage.bmp", pAvgTrainImg);
	// Create a large image made of many eigenobject images.
	// Must also convert each eigenobject image to a normal 8-bit UCHAR image instead of a 32-bit float image.
	printf("Saving the %d eigenvector images as 'out_eigenobjects.bmp'", nEigens);
	if (nEigens > 0) {
		// Put all the eigenobjects next to each other.
		int COLUMNS = 8;	// Put upto 8 images on a row.
		int nCols = min(nEigens, COLUMNS);
		int nRows = 1 + (nEigens / COLUMNS);	// Put the rest on new rows.
		int w = eigenVectArr[0]->width;
		int h = eigenVectArr[0]->height;
		CvSize size;
		size = cvSize(nCols * w, nRows * h);
		IplImage *bigImg = cvCreateImage(size, IPL_DEPTH_8U, 1);	// 8-bit Greyscale UCHAR image
		for (int i = 0; i < nEigens; i++) {
			// Get the eigenobject image.
			IplImage *byteImg = convertFloatImageToUcharImage(eigenVectArr[i]);
			// Paste it into the correct position.
			int x = w * (i % COLUMNS);
			int y = h * (i / COLUMNS);
			CvRect ROI = cvRect(x, y, w, h);
			cvSetImageROI(bigImg, ROI);
			cvCopyImage(byteImg, bigImg);
			cvResetImageROI(bigImg);
			cvReleaseImage(&byteImg);
		}
		cvSaveImage("out_eigenobjects.bmp", bigImg);
		cvReleaseImage(&bigImg);
	}
}

// Open the training data from the file 'objectdata.xml'.
int HaarTrainRecognition::loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interobject
	fileStorage = cvOpenFileStorage( "objectdata.xml", 0, CV_STORAGE_READ );
	if ( !fileStorage ) {
		printf("Can't open training database file 'objectdata.xml'.");
		return 0;
	}

	// Load the person names. Added by Shervin.
	personNames.clear();	// Make sure it starts as empty.
	nPersons = cvReadIntByName( fileStorage, 0, "nPersons", 0 );
	if (nPersons == 0) {
		printf("No people found in the training database 'objectdata.xml'.");
		return 0;
	}
	// Load each person's name.
	for (i = 0; i < nPersons; i++) {
		string sPersonName;
		char varname[200];
		sprintf( varname, "personName_%d", (i + 1) );
		sPersonName = cvReadStringByName(fileStorage, 0, varname );
		personNames.push_back( sPersonName );
	}

	// Load the data
	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainObjects = cvReadIntByName(fileStorage, 0, "nTrainObjects", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainObjectMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainObjectMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainObjects * sizeof(IplImage *));
	for (i = 0; i < nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	// release the file-storage interobject
	cvReleaseFileStorage( &fileStorage );

	printf("Training data loaded (%d training images of %d people):", nTrainObjects, nPersons);
	printf("People: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i = 1; i < nPersons; i++) {
		printf(", <%s>", personNames[i].c_str());
	}
	database_updated = true;
	return 1;
}

// Find the most likely person based on a detection. Returns the index, and stores the confidence value into pConfidence.
int HaarTrainRecognition::findNearestNeighbor(float * projectedTestObject, float *pConfidence)
{
	//double leastDistSq = 1e12;
	double leastDistSq = DBL_MAX;
	int i, iTrain, iNearest = 0;

	for (iTrain = 0; iTrain < nTrainObjects; iTrain++)
	{
		double distSq = 0;

		for (i = 0; i < nEigens; i++)
		{
			float d_i = projectedTestObject[i] - projectedTrainObjectMat->data.fl[iTrain * nEigens + i];
			#ifdef USE_MAHALANOBIS_DISTANCE
			distSq += d_i * d_i / eigenValMat->data.fl[i]; // Mahalanobis distance (might give better results than Eucalidean distance)
			#else
			distSq += d_i * d_i; // Euclidean distance.
			#endif
		}

		if (distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}

	// Return the confidence level based on the Euclidean distance,
	// so that similar images should give a confidence between 0.5 to 1.0,
	// and very different images should give a confidence between 0.0 to 0.5.
	*pConfidence = 1.0f - sqrt( leastDistSq / (float)(nTrainObjects * nEigens) ) / 255.0f;

	// Return the found index.
	return iNearest;
}





// Re-train the new object rec database
// Depending on the number of images in the training set and number of people, it might take 30 seconds or so.
// bool HaarTrainRecognition::retrainOnline(void)
// {
// 	// Free & Re-initialize the global variables.
// 	if (trainPersonNumMat)	{cvReleaseMat( &trainPersonNumMat ); trainPersonNumMat = 0;}
// 	int i;
// 	if (objectImgArr)
// 	{
// 		for (i = 0; i < nTrainObjects; i++)
// 			if (objectImgArr[i])	{cvReleaseImage( &objectImgArr[i] );}
// 		cvFree( &objectImgArr ); // array of object images
// 		objectImgArr = 0;
// 	}
// 	if (eigenVectArr)
// 	{
// 		for (i = 0; i < nEigens; i++)
// 			if (eigenVectArr[i])      {cvReleaseImage( &eigenVectArr[i] );}
// 		cvFree( &eigenVectArr ); // eigenvectors
// 		eigenVectArr = 0;
// 	}

// 	if (trainPersonNumMat) {cvReleaseMat( &trainPersonNumMat ); trainPersonNumMat = 0;} // array of person numbers
// 	personNames.clear();			// array of person names (indexed by the person number). Added by Shervin.
// 	nPersons = 0; // the number of people in the training set. Added by Shervin.
// 	nTrainObjects = 0; // the number of training images
// 	nEigens = 0; // the number of eigenvalues
// 	if (pAvgTrainImg) {cvReleaseImage( &pAvgTrainImg ); pAvgTrainImg = 0;} // the average image
// 	if (eigenValMat)  {cvReleaseMat( &eigenValMat ); eigenValMat = 0;} // eigenvalues
// 	if (projectedTrainObjectMat) {cvReleaseMat( &projectedTrainObjectMat ); projectedTrainObjectMat = 0;} // projected training objects
// 	// Retrain from the data in the files
// 	if (!learn("train.txt"))
// 		return (false);
// 	database_updated = true;
// 	return (true);

// }
