
// #include <opencv2/core.hpp>
#include <unistd.h>
#include "SURF_train_class.h"
#include <iostream>
#include "opencv2/calib3d.hpp"

//#define USE_MAHALANOBIS_DISTANCE	// You might get better recognition accuracy if you enable this.

SURFTrainClass::SURFTrainClass(const char *inputWorkingSpace) : minHessian(400) {

	writeWorkingSpace(inputWorkingSpace);	// write working space
}

SURFTrainClass::~SURFTrainClass(void) {

}

void SURFTrainClass::writeWorkingSpace(const char *inputWorkingSpace) {
	printf("Current working space path is: %s\n", inputWorkingSpace);
	chdir(inputWorkingSpace);
}

Mat SURFTrainClass::getSURFFeatureMat(Mat inputImage) {
	Mat grayImage;
	//Convert the RGB image obtained from camera into Grayscale
	cvtColor(inputImage, grayImage, CV_BGR2GRAY);
	//-- Step 1: Detect the keypoints using SURF Detector
	Ptr<SURF> detector = SURF::create( minHessian );
	std::vector<KeyPoint> keypoints;
	detector->detect( grayImage, keypoints);
	//-- Draw keypoints
	Mat outputImage;
	drawKeypoints( grayImage, keypoints, outputImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	return outputImage;
}

void SURFTrainClass::findMatches(Mat inputImage, const char *objectName) {
	//Convert the RGB image obtained from camera into Grayscale
	Mat grayImage;
	if(inputImage.channels() != 1 || inputImage.depth() != CV_8U)
	{
		cvtColor(inputImage, grayImage, CV_BGR2GRAY);
	}
	// -- Step 1: Detect the keypoints using SURF Detector
	char filename[20];
	sprintf(filename, "data/%s.pgm", objectName);
	Mat objectImage = imread( filename, IMREAD_GRAYSCALE );

	if ( !objectImage.data || !grayImage.data )
	{ printf(" --(!) Error reading images\n "); return; }

	//-- Step 1: Detect the keypoints using SURF Detector
	Ptr<SURF> detector = SURF::create( minHessian );

	std::vector<KeyPoint> objectKeypoints, sceneKeypoints;

	detector->detect( objectImage, objectKeypoints );
	detector->detect( grayImage, sceneKeypoints );

	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr<SURF> extractor = SURF::create();

	Mat objectDescriptors, sceneDescriptors;

	extractor->compute( objectImage, objectKeypoints, objectDescriptors );
	extractor->compute( grayImage, sceneKeypoints, sceneDescriptors );

	// //-- Step 3: Matching descriptor vectors using FLANN matcher
	// FlannBasedMatcher matcher;
	// std::vector< DMatch > matches;
	// matcher.match( objectDescriptors, sceneDescriptors, matches );

	// double max_dist = 0; double min_dist = 100;

	// //-- Quick calculation of max and min distances between keypoints
	// int limitAngle = 90;
	// for ( int i = 0; i < objectDescriptors.rows; i++ )
	// {	
	// 	double dist = matches[i].distance;
	// 	if ( dist < min_dist) {
	// 		if (matchInLimit(objectKeypoints[i], sceneKeypoints[i], limitAngle))
	// 			min_dist = dist;
	// 	}
	// 	if ( dist > max_dist ) max_dist = dist;
	// }

	// //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	// //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	// //-- small)
	// //-- PS.- radiusMatch can also be used here.
	// std::vector< DMatch > good_matches;
	// for ( int i = 0; i < objectDescriptors.rows; i++ )
	// {	
	// 	if ( matches[i].distance <= max(2 * min_dist, 0.02))
	// 	{	
	// 		good_matches.push_back( matches[i]);
	// 	}
	// }
	// if (good_matches.size() > 3) {
	// 	//-- Draw only "good" matches
	// 	Mat img_matches;
	// 	drawMatches( objectImage, objectKeypoints, inputImage, sceneKeypoints,
	// 	             good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	// 	             vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


	// 	//-- Localize the object
	// 	std::vector<Point2f> obj;
	// 	std::vector<Point2f> scene;

	// 	for ( int i = 0; i < good_matches.size(); i++ )
	// 	{
	// 		//-- Get the keypoints from the good matches
	// 		obj.push_back( objectKeypoints[ good_matches[i].queryIdx ].pt );
	// 		scene.push_back( sceneKeypoints[ good_matches[i].trainIdx ].pt );
	// 	}

	// 	Mat H = findHomography( obj, scene, RANSAC );

	// 	if (countNonZero(H) > 0) {
	// 		//-- Get the corners from the image_1 ( the object to be "detected" )
	// 		std::vector<Point2f> obj_corners(4);
	// 		obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint( objectImage.cols, 0 );
	// 		obj_corners[2] = cvPoint( objectImage.cols, objectImage.rows ); obj_corners[3] = cvPoint( 0, objectImage.rows );
	// 		std::vector<Point2f> scene_corners(4);

	// 		perspectiveTransform( obj_corners, scene_corners, H);
	// 		// -- Draw lines between the corners (the mapped object in the scene - image_2 )
	// 		line( img_matches, scene_corners[0] + Point2f( objectImage.cols, 0), scene_corners[1] + Point2f( objectImage.cols, 0), Scalar(0, 255, 0), 4 );
	// 		line( img_matches, scene_corners[1] + Point2f( objectImage.cols, 0), scene_corners[2] + Point2f( objectImage.cols, 0), Scalar( 0, 255, 0), 4 );
	// 		line( img_matches, scene_corners[2] + Point2f( objectImage.cols, 0), scene_corners[3] + Point2f( objectImage.cols, 0), Scalar( 0, 255, 0), 4 );
	// 		line( img_matches, scene_corners[3] + Point2f( objectImage.cols, 0), scene_corners[0] + Point2f( objectImage.cols, 0), Scalar( 0, 255, 0), 4 );

	// 		//-- Show detected matches
	// 		imshow( "Good Matches & Object detection", img_matches );

	// 		waitKey(0);
	// 	}
	// }
	////////////////////////////
	// NEAREST NEIGHBOR MATCHING USING FLANN LIBRARY (included in OpenCV)
	////////////////////////////
	cv::Mat results;
	cv::Mat dists;
	std::vector<std::vector<cv::DMatch> > matches;
	int k=2; // find the 2 nearest neighbors
	bool useBFMatcher = false; // SET TO TRUE TO USE BRUTE FORCE MATCHER
	if(objectDescriptors.type()==CV_8U)
	{
		// Binary descriptors detected (from ORB, Brief, BRISK, FREAK)
		printf("Binary descriptors detected...\n");
		if(useBFMatcher)
		{
			cv::BFMatcher matcher(cv::NORM_HAMMING); // use cv::NORM_HAMMING2 for ORB descriptor with WTA_K == 3 or 4 (see ORB constructor)
			matcher.knnMatch(objectDescriptors, sceneDescriptors, matches, k);
		}
		else
		{
			// Create Flann LSH index
			cv::flann::Index flannIndex(sceneDescriptors, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

			// search (nearest neighbor)
			flannIndex.knnSearch(objectDescriptors, results, dists, k, cv::flann::SearchParams() );
		}
	}
	else
	{
		// assume it is CV_32F
		printf("Float descriptors detected...\n");
		if(useBFMatcher)
		{
			cv::BFMatcher matcher(cv::NORM_L2);
			matcher.knnMatch(objectDescriptors, sceneDescriptors, matches, k);
		}
		else
		{
			// Create Flann KDTree index
			cv::flann::Index flannIndex(sceneDescriptors, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);

			// search (nearest neighbor)
			flannIndex.knnSearch(objectDescriptors, results, dists, k, cv::flann::SearchParams() );
		}
	}

	// Conversion to CV_32F if needed
	if(dists.type() == CV_32S)
	{
		cv::Mat temp;
		dists.convertTo(temp, CV_32F);
		dists = temp;
		printf("change to CV_32F");
	}

	////////////////////////////
	// PROCESS NEAREST NEIGHBOR RESULTS
	////////////////////////////
	// Find correspondences by NNDR (Nearest Neighbor Distance Ratio)
	float nndrRatio = 0.8f;
	std::vector<cv::Point2f> mpts_1, mpts_2; // Used for homography
	std::vector<int> indexes_1, indexes_2; // Used for homography
	std::vector<uchar> outlier_mask;  // Used for homography
	std::vector< DMatch > good_matches;
	// Check if this descriptor matches with those of the objects
	if(!useBFMatcher)
	{
		for(int i=0; i<objectDescriptors.rows; ++i)
		{
			// Apply NNDR
			//printf("q=%d dist1=%f dist2=%f\n", i, dists.at<float>(i,0), dists.at<float>(i,1));
			if(results.at<int>(i,0) >= 0 && results.at<int>(i,1) >= 0 &&
			   	dists.at<float>(i,0) <= nndrRatio * dists.at<float>(i,1))
			{
				mpts_1.push_back(objectKeypoints.at(i).pt);
				indexes_1.push_back(i);

				mpts_2.push_back(sceneKeypoints.at(results.at<int>(i,0)).pt);
				indexes_2.push_back(results.at<int>(i,0));
			}
		}
	}
	else
	{
		for(unsigned int i=0; i<matches.size(); ++i)
		{
			// Apply NNDR
			//printf("q=%d dist1=%f dist2=%f\n", matches.at(i).at(0).queryIdx, matches.at(i).at(0).distance, matches.at(i).at(1).distance);
			if(matches.at(i).size() == 2 &&
			   matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
			{
				mpts_1.push_back(objectKeypoints.at(matches.at(i).at(0).queryIdx).pt);
				indexes_1.push_back(matches.at(i).at(0).queryIdx);
				good_matches.push_back(matches.at(i).at(0));

				mpts_2.push_back(sceneKeypoints.at(matches.at(i).at(0).trainIdx).pt);
				indexes_2.push_back(matches.at(i).at(0).trainIdx);
			}
		}
	}

	// FIND HOMOGRAPHY
	unsigned int minInliers = 25;
	if(mpts_1.size() >= minInliers)
	{
		printf("find match, feature number: %d\n", (int)mpts_1.size());
		Mat img_matches = inputImage;
		// drawMatches( objectImage, objectKeypoints, inputImage, sceneKeypoints,
		//              good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		//              vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		Mat H = findHomography( mpts_1, mpts_2, RANSAC );

		if (countNonZero(H) > 0) {
			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint( objectImage.cols, 0 );
			obj_corners[2] = cvPoint( objectImage.cols, objectImage.rows ); obj_corners[3] = cvPoint( 0, objectImage.rows );
			std::vector<Point2f> scene_corners(4);

			perspectiveTransform( obj_corners, scene_corners, H);
			// -- Draw lines between the corners (the mapped object in the scene - image_2 )
			line( img_matches, scene_corners[0], scene_corners[1], Scalar( 0, 255, 0), 4 );
			line( img_matches, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
			line( img_matches, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
			line( img_matches, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );
			// put position on image
			int sceneObjectX = (scene_corners[0].x + scene_corners[1].x + scene_corners[2].x + scene_corners[3].x) / 4;
			int sceneObjectY = (scene_corners[0].y + scene_corners[1].y + scene_corners[2].y + scene_corners[3].y) / 4;
			printf("object x = %d, object y = %d", sceneObjectX, sceneObjectY);
			CvPoint sceneObjectCenter((scene_corners[0].x + scene_corners[3].x) / 2, sceneObjectY);
			char textCenter[20];
			sprintf(textCenter, "Center = (%d, %d)", sceneObjectX, sceneObjectY);
			putText(img_matches, textCenter, sceneObjectCenter, FONT_HERSHEY_PLAIN, 1, Scalar( 100, 255, 100));
			//-- Show detected matches
			imshow( "Good Matches & Object detection", img_matches );

			waitKey(0);
		}
	}
	else
	{
		printf("Not enough matches (%d) for homography...\n", (int)mpts_1.size());
	}
}

bool SURFTrainClass::matchInLimit(KeyPoint objectKeyPoint, KeyPoint imageKeyPoint, int angleRange) {
	int diff =  objectKeyPoint.angle - imageKeyPoint.angle;
	if (diff > -angleRange && diff < angleRange)
		return true;
	else
		return false;
}

