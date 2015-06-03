#include "framefeatures.h"

FrameFeatures::FrameFeatures(QObject *parent) :
    QThread(parent)
{

}

void FrameFeatures::run()
{
    found = false;
    dictionaryCreated = false;
    processFrames();
}

void FrameFeatures::processFrames()
{
    VideoCapture *capture  =  new cv::VideoCapture(filename);
    int numberOfFrames = capture->get(CV_CAP_PROP_FRAME_COUNT);
    int frameRate = (int) capture->get(CV_CAP_PROP_FPS);
    int Nth = frameRate/2;
    int j = 0;
    Mat frm;
    capture->read(frm); // get a new frame from camera

    vector<KeyPoint> keypoints_scene;

    Mat descriptors_scene;

    //Ptr<FeatureDetector> detector =  new PyramidAdaptedFeatureDetector( new DynamicAdaptedFeatureDetector ( new SurfAdjuster(700,true), 500, 1000, 3), 4);
    //Ptr<FeatureDetector> detector = FeatureDetector::create("MSER");
    //Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");

        //SIFT
        //Ptr<FeatureDetector> detector = new cv::SiftFeatureDetector();
        //Ptr<DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();

        //ORB
        //cv::FeatureDetector* featureDetector = new OrbFeatureDetector();
        //cv::DescriptorExtractor* descriptorExtractor = new OrbDescriptorExtractor();

        //FAST + SIFT
        //cv::FeatureDetector* featureDetector = new FastFeatureDetector();
        //cv::DescriptorExtractor* descriptorExtractor = new cv::SiftDescriptorExtractor();

        //MSER + SIFT
        Ptr<FeatureDetector> detector = new cv::MserFeatureDetector();
        Ptr<DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();

        //MSER + FREAK
        //Ptr<FeatureDetector> detector = new cv::MserFeatureDetector();
       // Ptr<DescriptorExtractor> extractor = new cv::FREAK;

        //SURF
        //Ptr<FeatureDetector> detector = new cv::SurfFeatureDetector();
        //Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();

        //SURF + FREAK
        //cv::FeatureDetector* featureDetector = new cv::SurfFeatureDetector();
        //cv::DescriptorExtractor *descriptorExtractor = new cv::FREAK();

        //KAZE
        //cv::FeatureDetector* featureDetector = new cv::KazeFeatureDetector();
        //cv::DescriptorExtractor* descriptorExtractor = new cv::KazeDescriptorExtractor();


    while( j < (numberOfFrames-Nth+1) && !frm.empty() )
    {
        keypoints_scene.clear();
        cv::Mat mask(0, 0, CV_8UC1);
        //cvtColor(frm, frm, CV_BGR2GRAY);

        //-- Step 1: Detect the keypoints using Detector
        detector->detect(frm, keypoints_scene, mask);

        for (int k = 0; k < keypoints_scene.size(); k++)
        {
            float angle = keypoints_scene.at(k).angle;
            float size = keypoints_scene.at(k).size;
        }

        //-- Step 2: Calculate descriptors (feature vectors)
        extractor->compute(frm, keypoints_scene, descriptors_scene);

        //-- Passing values to mainwindow.cpp
        frameVector.push_back(frm);
        keypoints_frameVector.push_back(keypoints_scene);
        descriptors_sceneVector.push_back(descriptors_scene);

        qDebug() << "Size of keypoints_scene bin = " << keypoints_scene.size();
        qDebug() << "Size of Scene Descriptor:  " << descriptors_sceneVector.size() << ": " << descriptors_scene.rows << " x " << descriptors_scene.cols;

        if (!descriptors_scene.empty())
        {
            bowTrainer->add(descriptors_scene);
        }

        j += Nth;
        capture->set(CV_CAP_PROP_POS_FRAMES, j);
        capture->read(frm);// get every Nth frame (every second of video)
        found = true;
    }

    calculateCluster();

    emit onFeaturesFound(found);
    emit onDictionaryMade(dictionaryCreated);
}

void FrameFeatures::setFilename(string filename)
{
    this->filename = filename;
}

vector<cv::Mat > FrameFeatures::getFeatureVectors()
{
    return this->descriptors_sceneVector;
}

void FrameFeatures::calculateCluster()
{
    /************* TRAINING VOCABULARY **************/
    //Training the Bag of Words model with the selected feature components
    vector<Mat> descriptors = bowTrainer->getDescriptors();

    int count=0;
    for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
    {
        count+=iter->rows;
    }
    qDebug() << "Clustering " << count << " features" << endl;

    if (count > DICTIONARY_SIZE)
    {
        dictionary = bowTrainer->cluster();

        if (!dictionary.empty())
        {
            dictionaryCreated = true;
        }

        qDebug() << "dictionary size: "<< dictionary.rows << " x " << dictionary.cols;
    }
}

