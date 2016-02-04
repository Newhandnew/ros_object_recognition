#include <cvaux.h>
#include "haar_train_recognition.h"

//#define USE_MAHALANOBIS_DISTANCE	// You might get better recognition accuracy if you enable this.

HaarTrainRecognition::HaarTrainRecognition() {
	SAVE_EIGENFACE_IMAGES = 1;
	faceImgArr = 0;
	faceWidth = 120;
	faceHeight = 90;
	nPersons = 0;
	nTrainFaces = 0;
	nEigens = 0;
	pAvgTrainImg = 0;
	eigenVectArr = 0;
	eigenValMat = 0;
	projectedTrainFaceMat = 0;
	database_updated = false;

	// Load the previously saved training data
	trainPersonNumMat = 0;
	if ( loadTrainingData( &trainPersonNumMat ) )
	{
		faceWidth = pAvgTrainImg->width;
		faceHeight = pAvgTrainImg->height;
		database_updated = true;
	}
	else
	{
		printf("Will try to train from images");
		if (!retrainOnline())
			printf("Could not train from images");

	}
}
HaarTrainRecognition::~HaarTrainRecognition(void) {
	cvReleaseHaarClassifierCascade( &faceCascade );
	if (trainPersonNumMat)	cvReleaseMat( &trainPersonNumMat );
	int i;
	if (faceImgArr)
	{
		for (i = 0; i < nTrainFaces; i++)
			if (faceImgArr[i])	cvReleaseImage( &faceImgArr[i] );
		cvFree( &faceImgArr );
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
	if (projectedTrainFaceMat) cvReleaseMat( &projectedTrainFaceMat );
}

void HaarTrainRecognition::writeWorkingSpace(char *inputWorkingSpace) {
	*workingSpacePath = *inputWorkingSpace;
	printf("Current working space path is: %s", workingSpacePath);
}

// Perform face detection on the input image, using the given Haar cascade classifier.
// Returns a rectangle for the detected region in the given image.
CvRect HaarTrainRecognition::detectFaceInImage(const IplImage *inputImg, const CvHaarClassifierCascade* cascade )
{
	const CvSize minFeatureSize = cvSize(20, 20);
	const int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH;	// Only search for 1 face.
	const float search_scale_factor = 1.1f;
	IplImage *detectImg;
	IplImage *greyImg = 0;
	CvMemStorage* storage;
	CvRect rc;
	//double t;
	CvSeq* rects;


	storage = cvCreateMemStorage(0);
	cvClearMemStorage( storage );

	// If the image is color, use a greyscale copy of the image.
	detectImg = (IplImage*)inputImg;	// Assume the input image is to be used.
	if (inputImg->nChannels > 1)
	{
		greyImg = cvCreateImage(cvSize(inputImg->width, inputImg->height), IPL_DEPTH_8U, 1 );
		cvCvtColor( inputImg, greyImg, CV_BGR2GRAY );
		detectImg = greyImg;	// Use the greyscale version as the input.
	}

	// Detect all the faces.
	//$$$$$t = (double)cvGetTickCount();
	rects = cvHaarDetectObjects( detectImg, (CvHaarClassifierCascade*)cascade, storage,
	                             search_scale_factor, 3, flags, minFeatureSize );
	//$$$$$t = (double)cvGetTickCount() - t;
	//$$$$$printf("[Face Detection took %d ms and found %d objects]\n", cvRound( t/((double)cvGetTickFrequency()*1000.0) ), rects->total );

	// Get the first detected face (the biggest).
	if (rects->total > 0)
	{
		rc = *(CvRect*)cvGetSeqElem( rects, 0 );
	}
	else
		rc = cvRect(-1, -1, -1, -1);	// Couldn't find the face.

	//cvReleaseHaarClassifierCascade( &cascade );
	//cvReleaseImage( &detectImg );
	if (greyImg)
		cvReleaseImage( &greyImg );
	cvReleaseMemStorage( &storage );

	return rc;	// Return the biggest face found, or (-1,-1,-1,-1).
}



// Return a new image that is always greyscale, whether the input image was RGB or Greyscale.
// Remember to free the returned image using cvReleaseImage() when finished.
IplImage* HaarTrainRecognition::convertImageToGreyscale(const IplImage *imageSrc)
{
	IplImage *imageGrey;
	// Either convert the image to greyscale, or make a copy of the existing greyscale image.
	// This is to make sure that the user can always call cvReleaseImage() on the output, whether it was greyscale or not.
	if (imageSrc->nChannels == 3) {
		imageGrey = cvCreateImage( cvGetSize(imageSrc), IPL_DEPTH_8U, 1 );
		cvCvtColor( imageSrc, imageGrey, CV_BGR2GRAY );
	}
	else {
		imageGrey = cvCloneImage(imageSrc);
	}
	return imageGrey;
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

// Returns a new image that is a cropped version of the original image.
IplImage* HaarTrainRecognition::cropImage(const IplImage *img, const CvRect region)
{
	IplImage *imageTmp;
	IplImage *imageRGB;
	CvSize size;
	size.height = img->height;
	size.width = img->width;

	if (img->depth != IPL_DEPTH_8U) {
		printf("ERROR in cropImage: Unknown image depth of %d given in cropImage() instead of 8 bits per pixel.", img->depth);
		exit(1);
	}

	// First create a new (color or greyscale) IPL Image and copy contents of img into it.
	imageTmp = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(img, imageTmp, NULL);

	// Create a new image of the detected region
	// Set region of interest to that surrounding the face
	cvSetImageROI(imageTmp, region);
	// Copy region of interest (i.e. face) into a new iplImage (imageRGB) and return it
	size.width = region.width;
	size.height = region.height;
	imageRGB = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(imageTmp, imageRGB, NULL);	// Copy just the region.

	cvReleaseImage( &imageTmp );
	return imageRGB;
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




void HaarTrainRecognition::storeEigenfaceImages()
{
	// Store the average image to a file
	printf("Saving the image of the average face as 'out_averageImage.bmp'.");
	cvSaveImage("out_averageImage.bmp", pAvgTrainImg);
	// Create a large image made of many eigenface images.
	// Must also convert each eigenface image to a normal 8-bit UCHAR image instead of a 32-bit float image.
	printf("Saving the %d eigenvector images as 'out_eigenfaces.bmp'", nEigens);
	if (nEigens > 0) {
		// Put all the eigenfaces next to each other.
		int COLUMNS = 8;	// Put upto 8 images on a row.
		int nCols = min(nEigens, COLUMNS);
		int nRows = 1 + (nEigens / COLUMNS);	// Put the rest on new rows.
		int w = eigenVectArr[0]->width;
		int h = eigenVectArr[0]->height;
		CvSize size;
		size = cvSize(nCols * w, nRows * h);
		IplImage *bigImg = cvCreateImage(size, IPL_DEPTH_8U, 1);	// 8-bit Greyscale UCHAR image
		for (int i = 0; i < nEigens; i++) {
			// Get the eigenface image.
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
		cvSaveImage("out_eigenfaces.bmp", bigImg);
		cvReleaseImage(&bigImg);
	}
}

// Train from the data in the given text file, and store the trained data into the file 'facedata.xml'.
bool HaarTrainRecognition::learn(const char *szFileTrain)
{
	int i, offset;

	// load training data
	printf("Loading the training images in '%s'", szFileTrain);
	nTrainFaces = loadFaceImgArray(szFileTrain);
	printf("Got %d training images.\n", nTrainFaces);
	if ( nTrainFaces < 2 )
	{
		fprintf(stderr,
		        "Need 2 or more training faces"
		        "Input file contains only %d", nTrainFaces);
		return false;
	}

	// do PCA on the training faces
	doPCA();

	// project the training images onto the PCA subspace
	projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
	offset = projectedTrainFaceMat->step / sizeof(float);
	for (i = 0; i < nTrainFaces; i++)
	{
		//int offset = i * nEigens;
		cvEigenDecomposite(
		    faceImgArr[i],
		    nEigens,
		    eigenVectArr,
		    0, 0,
		    pAvgTrainImg,
		    //projectedTrainFaceMat->data.fl + i*nEigens);
		    projectedTrainFaceMat->data.fl + i * offset);
	}

	// store the recognition data as an xml file
	storeTrainingData();

	// Save all the eigenvectors as images, so that they can be checked.
	//if (SAVE_EIGENFACE_IMAGES) {
	//	storeEigenfaceImages();
	//}
	return true;

}


// Open the training data from the file 'facedata.xml'.
int HaarTrainRecognition::loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	if ( !fileStorage ) {
		printf("Can't open training database file 'facedata.xml'.");
		return 0;
	}

	// Load the person names. Added by Shervin.
	personNames.clear();	// Make sure it starts as empty.
	nPersons = cvReadIntByName( fileStorage, 0, "nPersons", 0 );
	if (nPersons == 0) {
		printf("No people found in the training database 'facedata.xml'.");
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
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces * sizeof(IplImage *));
	for (i = 0; i < nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );

	printf("Training data loaded (%d training images of %d people):", nTrainFaces, nPersons);
	printf("People: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i = 1; i < nPersons; i++) {
		printf(", <%s>", personNames[i].c_str());
	}
	database_updated = true;
	return 1;
}


// Save the training data to the file 'facedata.xml'.
void HaarTrainRecognition::storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

	// Store the person names. Added by Shervin.
	cvWriteInt( fileStorage, "nPersons", nPersons );
	for (i = 0; i < nPersons; i++) {
		char varname[200];
		sprintf( varname, "personName_%d", (i + 1) );
		cvWriteString(fileStorage, varname, personNames[i].c_str(), 0);
	}

	// store all the data
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	cvWrite(fileStorage, "trainPersonNumMat", trainPersonNumMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0, 0));
	for (i = 0; i < nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0, 0));
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );
}

// Find the most likely person based on a detection. Returns the index, and stores the confidence value into pConfidence.
int HaarTrainRecognition::findNearestNeighbor(float * projectedTestFace, float *pConfidence)
{
	//double leastDistSq = 1e12;
	double leastDistSq = DBL_MAX;
	int i, iTrain, iNearest = 0;

	for (iTrain = 0; iTrain < nTrainFaces; iTrain++)
	{
		double distSq = 0;

		for (i = 0; i < nEigens; i++)
		{
			float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain * nEigens + i];
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
	*pConfidence = 1.0f - sqrt( leastDistSq / (float)(nTrainFaces * nEigens) ) / 255.0f;

	// Return the found index.
	return iNearest;
}

// Do the Principal Component Analysis, finding the average image
// and the eigenfaces that represent any image in the given dataset.
void HaarTrainRecognition::doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;

	// set the number of eigenvalues to use
	nEigens = nTrainFaces - 1;

	// allocate the eigenvector images
	faceImgSize.width  = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for (i = 0; i < nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// allocate the eigenvalue array
	eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// allocate the averaged image
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// set the PCA termination criterion
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);
	printf("**** nTrainFaces: %d", nTrainFaces);
	// compute average image, eigenvalues, and eigenvectors
	cvCalcEigenObjects(
	    nTrainFaces,
	    (void*)faceImgArr,
	    (void*)eigenVectArr,
	    CV_EIGOBJ_NO_CALLBACK,
	    0,
	    0,
	    &calcLimit,
	    pAvgTrainImg,
	    eigenValMat->data.fl);

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

// Read the names & image filenames of people from a text file, and load all those images listed.
int HaarTrainRecognition::loadFaceImgArray(const char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces = 0;
	int i;
	IplImage *pfaceImg;
	IplImage *psizedImg;
	IplImage *pequalizedImg;
	// open the input file
	if ( !(imgListFile = fopen(filename, "r")) )
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// count the number of faces
	while ( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	// allocate the face-image array and person number matrix
	faceImgArr        = (IplImage **)cvAlloc( nFaces * sizeof(IplImage *) );
	trainPersonNumMat = cvCreateMat( 1, nFaces, CV_32SC1 );

	personNames.clear();	// Make sure it starts as empty.
	nPersons = 0;

	// store the face images in an array
	for (iFace = 0; iFace < nFaces; iFace++)
	{
		char personName[256];
		string sPersonName;
		int personNumber;
		// read person number (beginning with 1), their name and the image filename.
		fscanf(imgListFile, "%d %s %s", &personNumber, personName, imgFilename);
		sPersonName = personName;
		//printf("Got %d: %d, <%s>, <%s>.\n", iFace, personNumber, personName, imgFilename);

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
		trainPersonNumMat->data.i[iFace] = personNumber;

		// load the face image
		pfaceImg = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);
		psizedImg = resizeImage(pfaceImg, faceWidth, faceHeight);
		// Give the image a standard brightness and contrast, in case it was too dark or low contrast.
		pequalizedImg = cvCreateImage(cvGetSize(psizedImg), 8, 1);	// Create an empty greyscale image
		cvEqualizeHist(psizedImg, pequalizedImg);
		faceImgArr[iFace] = pequalizedImg;
		cvReleaseImage( &pfaceImg ); cvReleaseImage( &psizedImg );
		if ( !faceImgArr[iFace] )
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
	}

	fclose(imgListFile);

	printf("Data loaded from '%s': (%d images of %d people).\n", filename, nFaces, nPersons);
	printf("People: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i = 1; i < nPersons; i++) {
		printf(", <%s>", personNames[i].c_str());
	}
	printf(".\n");

	return nFaces;
}



// Re-train the new face rec database
// Depending on the number of images in the training set and number of people, it might take 30 seconds or so.
bool HaarTrainRecognition::retrainOnline(void)
{
	// Free & Re-initialize the global variables.
	if (trainPersonNumMat)	{cvReleaseMat( &trainPersonNumMat ); trainPersonNumMat = 0;}
	int i;
	if (faceImgArr)
	{
		for (i = 0; i < nTrainFaces; i++)
			if (faceImgArr[i])	{cvReleaseImage( &faceImgArr[i] );}
		cvFree( &faceImgArr ); // array of face images
		faceImgArr = 0;
	}
	if (eigenVectArr)
	{
		for (i = 0; i < nEigens; i++)
			if (eigenVectArr[i])      {cvReleaseImage( &eigenVectArr[i] );}
		cvFree( &eigenVectArr ); // eigenvectors
		eigenVectArr = 0;
	}

	if (trainPersonNumMat) {cvReleaseMat( &trainPersonNumMat ); trainPersonNumMat = 0;} // array of person numbers
	personNames.clear();			// array of person names (indexed by the person number). Added by Shervin.
	nPersons = 0; // the number of people in the training set. Added by Shervin.
	nTrainFaces = 0; // the number of training images
	nEigens = 0; // the number of eigenvalues
	if (pAvgTrainImg) {cvReleaseImage( &pAvgTrainImg ); pAvgTrainImg = 0;} // the average image
	if (eigenValMat)  {cvReleaseMat( &eigenValMat ); eigenValMat = 0;} // eigenvalues
	if (projectedTrainFaceMat) {cvReleaseMat( &projectedTrainFaceMat ); projectedTrainFaceMat = 0;} // projected training faces
	// Retrain from the data in the files
	if (!learn("train.txt"))
		return (false);
	database_updated = true;
	return (true);

}
