#include <object_recognition.h>

ObjectRecognition::ObjectRecognition(): trainFileName("pattern.txt") {
    ros::NodeHandle n;
    n.param<bool>("object_recognition/show_screen_flag", fShowScreen, true);
    match_publisher = n.advertise<object_recognition::match_data_array>("matches", 10);
    service_recognition = n.advertiseService("recognition", &ObjectRecognition::recognitionCB, this);
    service_save = n.advertiseService("save_object", &ObjectRecognition::saveObjectCB, this);
    image_processing = new ImageProcessing("object_recognition");
    image_processing->setFlagShowScreen(fShowScreen);
    trainModel = new SURFTrainClass(image_processing->getWorkingSpacePath(), fShowScreen);
}

void ObjectRecognition::publishMatchObjects(vector<SURFTrainClass::matchData> matchedObjects) {
    object_recognition::match_data data;
    object_recognition::match_data_array msg;
    int i;
    for (i = 0; i < matchedObjects.size(); i++) {
        string name(matchedObjects[i].name);
        data.name = name;
        data.x = matchedObjects[i].x;
        data.y = matchedObjects[i].y;
        msg.objects.push_back(data);
    }
    match_publisher.publish(msg);
}

bool ObjectRecognition::recognitionCB(object_recognition::recognition::Request  &req, 
    object_recognition::recognition::Response &res) {
    matchedObjects = trainModel->findMatches(image_processing->getObjectImage(), trainFileName);
    object_recognition::match_data data;
    int i;
    for (i = 0; i < matchedObjects.size(); i++) {
        string name(matchedObjects[i].name);
        data.name = name;
        data.x = matchedObjects[i].x;
        data.y = matchedObjects[i].y;
        res.objects.push_back(data);
    }
    return true;
}

bool ObjectRecognition::saveObjectCB(object_recognition::save_object::Request  &req, 
    object_recognition::save_object::Response &res) {
    image_processing->saveObjectImages(req.objectname.c_str());
    return true;
}

void ObjectRecognition::run() {
    int i;
    ros::Rate r(20); // 10 hz
    while(ros::ok()) {
        ros::spinOnce();
        image_processing->showCombineImages();
        char keyInput = cvWaitKey(1);
        switch(keyInput) {
            case 'o':   // object
                image_processing->showMaxObjectImage();
                break;
            case 's':   // save
                image_processing->saveObjectImages("object");
                break;
            case 'r':   // recognition
                matchedObjects = trainModel->findMatches(image_processing->getObjectImage(), trainFileName);
                break;
            case 'f':   //feature
                featureImage = trainModel->getSURFFeatureMat(image_processing->getObjectImage());
                image_processing->showImage(featureImage);
                break;
            case 'p':   // print
                for (i = 0; i < matchedObjects.size(); i++) {
                    printf("match: %s, x = %d, y = %d\n", matchedObjects[i].name, matchedObjects[i].x, matchedObjects[i].y);
                }
                publishMatchObjects(matchedObjects);
                break;
            case 27:    // ESC = 17
                exit(1);
                break;
        }
        // object_recognition.saveTrainingSet("object");
        r.sleep();
    }
}

void ObjectRecognition::close() {
    delete image_processing;
    delete trainModel;
}


//  ====================================================================
int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_recognition");
    ObjectRecognition object_recognition_node;
    object_recognition_node.run();
    object_recognition_node.close();
    return 0;
}
