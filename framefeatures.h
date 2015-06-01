#ifndef FRAMEFEATURES_H
#define FRAMEFEATURES_H

#define DICTIONARY_SIZE 1500

#include <QThread>
#include <QtCore>
#include <QDebug>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/video/video.hpp>
 #include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

class FrameFeatures : public QThread
{
    Q_OBJECT
public:
    explicit FrameFeatures(QObject *parent = 0);
    void run();
    void processFrames();
    void setFilename(string filename);
    vector<cv::Mat > getFeatureVectors();
    void calculateCluster();

    Mat dictionary;
    BOWKMeansTrainer* bowTrainer =  new BOWKMeansTrainer(DICTIONARY_SIZE, TermCriteria(CV_TERMCRIT_ITER, 10, 0.001), 1, KMEANS_PP_CENTERS); //Construct BOWKMeansTrainer
    Ptr<DescriptorExtractor> extractor;
    vector<cv::Mat> frameVector;
    vector<vector<KeyPoint> > keypoints_frameVector;
    vector<Mat> histogram_sceneVector;
    vector<cv::Mat > descriptors_sceneVector;
    string filename;
    bool found;    
    bool dictionaryCreated;

signals:
    void onFeaturesFound(const bool &found);
    void onDictionaryMade(const bool &dictionary);

public slots:

};

#endif // FRAMEFEATURES_H
