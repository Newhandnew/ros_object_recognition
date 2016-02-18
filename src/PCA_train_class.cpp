#include <cvaux.h>
#include "PCA_train_class.h"
#include <unistd.h>

//#define USE_MAHALANOBIS_DISTANCE	// You might get better recognition accuracy if you enable this.

PCATrainClass::PCATrainClass(const char *inputWorkingSpace) {
	fSaveEigenImage = 1;		// Set to 0 if you don't want images of the Eigenvectors saved to files (for debugging).
	objectImgArr = 0;
	// objectWidth = 120;
	// objectHeight = 90;
	nObjects = 0;
	nTrainObjects = 0;
	nEigens = 0;
	pAvgTrainImg = 0;
	eigenVectArr = 0;
	eigenValMat = 0;
	projectedTrainObjectMat = 0;
	database_updated = false;

	writeWorkingSpace(inputWorkingSpace);	// write working space
	// Load the previously saved training data
	trainObjectNumMat = 0;
	loadTrainingData(&trainObjectNumMat);

}

PCATrainClass::~PCATrainClass(void) {
	// cvReleaseHaarClassifierCascade( &objectCascade );
	if (trainObjectNumMat)	cvReleaseMat( &trainObjectNumMat );
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
	objectNames.clear();
	if (pAvgTrainImg) cvReleaseImage( &pAvgTrainImg );
	if (eigenValMat)  cvReleaseMat( &eigenValMat );
	if (projectedTrainObjectMat) cvReleaseMat( &projectedTrainObjectMat );
}

void PCATrainClass::writeWorkingSpace(const char *inputWorkingSpace) {
	printf("Current working space path is: %s\n", inputWorkingSpace);
	chdir(inputWorkingSpace);
}


// Creates a new image copy that is of a desired size.
// Remember to free the new image later.
IplImage* PCATrainClass::resizeImage(const IplImage *origImg, int newWidth, int newHeight)
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
IplImage* PCATrainClass::convertFloatImageToUcharImage(const IplImage *srcImg)
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
bool PCATrainClass::learn(const char *szFileTrain)
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
int PCATrainClass::loadObjectImgArray(const char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iPicture, nPictures = 0;
	int i;
	IplImage *pObjectImg;
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
	while ( fgets(imgFilename, 512, imgListFile) ) ++nPictures;
	rewind(imgListFile);

	// allocate the object-image array and object number matrix
	objectImgArr        = (IplImage **)cvAlloc( nPictures * sizeof(IplImage *) );
	trainObjectNumMat = cvCreateMat( 1, nPictures, CV_32SC1 );

	objectNames.clear();	// Make sure it starts as empty.
	nObjects = 0;

	// store the object images in an array
	for (iPicture = 0; iPicture < nPictures; iPicture++)
	{
		char objectName[256];
		string sPersonName;
		int objectNumber;
		// read object number (beginning with 1), their name and the image filename.
		fscanf(imgListFile, "%d %s %s", &objectNumber, objectName, imgFilename);
		sPersonName = objectName;
		//printf("Got %d: %d, <%s>, <%s>.\n", iPicture, objectNumber, objectName, imgFilename);

		// Check if a new object is being loaded.
		if (objectNumber > nObjects) {
			// Allocate memory for the extra object (or possibly multiple), using this new object's name.
			for (i = nObjects; i < objectNumber; i++) {
				objectNames.push_back( sPersonName );
			}
			nObjects = objectNumber;
			//printf("Got new object <%s> -> nObjects = %d [%d]\n", sPersonName.c_str(), nObjects, objectNames.size());
		}

		// Keep the data
		trainObjectNumMat->data.i[iPicture] = objectNumber;

		// load the object image
		pObjectImg = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);
		if(iPicture == 0) {
			imgWidth = pObjectImg->width;
			imgHeight = pObjectImg->height;
		}
		psizedImg = resizeImage(pObjectImg, imgWidth, imgHeight);
		// Give the image a standard brightness and contrast, in case it was too dark or low contrast.
		pequalizedImg = cvCreateImage(cvGetSize(psizedImg), 8, 1);	// Create an empty greyscale image
		cvEqualizeHist(psizedImg, pequalizedImg);
		objectImgArr[iPicture] = pequalizedImg;
		cvReleaseImage( &psizedImg ); cvReleaseImage( &pObjectImg );
		if ( !objectImgArr[iPicture] )
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
	}

	fclose(imgListFile);

	printf("Data loaded from '%s': (%d images of %d objects).\n", filename, nPictures, nObjects);
	printf("object: ");
	if (nObjects > 0)
		printf("<%s>", objectNames[0].c_str());
	for (i = 1; i < nObjects; i++) {
		printf(", <%s>", objectNames[i].c_str());
	}
	printf(".\n");

	return nPictures;
}


// Do the Principal Component Analysis, finding the average image
// and the eigenobjects that represent any image in the given dataset.
void PCATrainClass::doPCA()
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
	printf("**** nTrainObjects: %d\n", nTrainObjects);
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
void PCATrainClass::storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interobject
	fileStorage = cvOpenFileStorage( "objectdata.xml", 0, CV_STORAGE_WRITE );

	// Store the object names. Added by Shervin.
	cvWriteInt( fileStorage, "nObjects", nObjects );
	for (i = 0; i < nObjects; i++) {
		char varname[200];
		sprintf( varname, "objectName_%d", (i + 1) );
		cvWriteString(fileStorage, varname, objectNames[i].c_str(), 0);
	}

	// store all the data
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainObjects", nTrainObjects );
	cvWrite(fileStorage, "trainObjectNumMat", trainObjectNumMat, cvAttrList(0, 0));
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

void PCATrainClass::storeEigenobjectImages()
{
	// Store the average image to a file
	printf("Saving the image of the average object as 'out_averageImage.bmp'.\n");
	cvSaveImage("out_averageImage.bmp", pAvgTrainImg);
	// Create a large image made of many eigenobject images.
	// Must also convert each eigenobject image to a normal 8-bit UCHAR image instead of a 32-bit float image.
	printf("Saving the %d eigenvector images as 'out_eigenobjects.bmp'\n", nEigens);
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
int PCATrainClass::loadTrainingData(CvMat ** ptrainObjectNumMat)
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interobject
	fileStorage = cvOpenFileStorage( "objectdata.xml", 0, CV_STORAGE_READ );
	if ( !fileStorage ) {
		printf("Can't open training database file 'objectdata.xml'.");
		return 0;
	}

	// Load the object names. Added by Shervin.
	objectNames.clear();	// Make sure it starts as empty.
	nObjects = cvReadIntByName( fileStorage, 0, "nObjects", 0 );
	if (nObjects == 0) {
		printf("No object found in the training database 'objectdata.xml'.");
		return 0;
	}
	// Load each object's name.
	for (i = 0; i < nObjects; i++) {
		string sPersonName;
		char varname[200];
		sprintf( varname, "objectName_%d", (i + 1) );
		sPersonName = cvReadStringByName(fileStorage, 0, varname );
		objectNames.push_back( sPersonName );
	}

	// Load the data
	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainObjects = cvReadIntByName(fileStorage, 0, "nTrainObjects", 0);
	*ptrainObjectNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainObjectNumMat", 0);
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

	printf("Training data loaded (%d training images of %d object):", nTrainObjects, nObjects);
	if (nObjects > 0)
		printf("<%s>", objectNames[0].c_str());
	for (i = 1; i < nObjects; i++) {
		printf(", <%s>", objectNames[i].c_str());
	}
	printf("\n");
	database_updated = true;
	return 1;
}

// Find the most likely object based on a detection. Returns the index, and stores the confidence value into pConfidence.
int PCATrainClass::findNearestNeighbor(IplImage *inputImage, float *pConfidence)
{
	//double leastDistSq = 1e12;
	double leastDistSq = DBL_MAX;
	int i, iTrain, iNearest = 0;
	IplImage *grayImg = cvCreateImage(cvGetSize(inputImage), 8, 1);	// Create an empty greyscale image
	IplImage *equalizedImg = cvCreateImage(cvGetSize(inputImage), 8, 1);	// Create an empty greyscale image
	IplImage *psizedImg;
	float * projectedTestObject = 0;
	int objectIndex;

	if(nEigens < 1) 
    {
        printf("NO database available, goal is Aborted");
        cvReleaseImage(&equalizedImg);
        return -1;
    }
    cvCvtColor(inputImage, grayImg, CV_RGB2GRAY);
	cvEqualizeHist(grayImg, equalizedImg);	// equalized image
	psizedImg = resizeImage(equalizedImg, pAvgTrainImg -> width, pAvgTrainImg -> height);	// resize image
    // Project the test images onto the PCA subspace
    projectedTestObject = (float *)cvAlloc( nEigens * sizeof(float));
    // project the test image onto the PCA subspace
    cvEigenDecomposite(psizedImg, nEigens, eigenVectArr, 0, 0, pAvgTrainImg, projectedTestObject);
	
    // printf("nEigens: %d\n", nEigens);

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
			// printf("i =  %d, distSq = %f\n", i, distSq);
		}
		// get the least distance index
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
	objectIndex  = trainObjectNumMat->data.i[iNearest];
	return objectIndex;
}





// Re-train the new object rec database
// Depending on the number of images in the training set and number of people, it might take 30 seconds or so.
// bool PCATrainClass::retrainOnline(void)
// {
// 	// Free & Re-initialize the global variables.
// 	if (trainObjectNumMat)	{cvReleaseMat( &trainObjectNumMat ); trainObjectNumMat = 0;}
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

// 	if (trainObjectNumMat) {cvReleaseMat( &trainObjectNumMat ); trainObjectNumMat = 0;} // array of object numbers
// 	objectNames.clear();			// array of object names (indexed by the object number). Added by Shervin.
// 	nObjects = 0; // the number of people in the training set. Added by Shervin.
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
