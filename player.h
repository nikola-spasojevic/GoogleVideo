#ifndef PLAYER_H
#define PLAYER_H

#include <QMouseEvent>
#include <QMutex>
#include <QImage>
#include <QWaitCondition>
#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QTime>
#include "framefeatures.h"
#include "blob.h"

class QLabel;

class Player : public QThread
{    Q_OBJECT
 private:
    bool stop;
    QMutex mutex;
    QWaitCondition condition;
    Mat frame;
    Mat RGBframe;
    Mat processedFrame;
    int frameRate;
    VideoCapture *capture;
    QImage img;
    QImage imgProcessed;

 signals:
    void originalImage(const QImage &image);
    void processedImage(const QImage &imgProcessed);
    void dictionaryPassed(const bool &dictionary);

 public slots:
    void onFeaturesPassed(bool found);
    void onDictionaryPassed(bool dictionary);

 protected:
     void run();

 public:
    Player(QObject *parent = 0);
    ~Player();
    bool loadVideo(string filename);
    void Play();
    void Stop();
    bool isStopped() const;
    void msleep(int ms);
    void setCurrentFrame(int frameNumber);
    double getFrameRate();
    double getCurrentFrame();
    double getNumberOfFrames();
    void ProcessFrame();
    void Tracking();
    void getFeatureHeatMap();
    vector<cv::Mat > featureVectorPerFrame;
    Mat dictionary;
    FrameFeatures *frameFeatures;

    /**************** Tracking ****************/
    cv::Mat rawFrame,rawCopyFrame,foregroundFrame, foregroundFrameBuffer, roiFrame, roiFrameBuffer, hsvRoiFrame, roiFrameMask;
    cv::BackgroundSubtractorMOG2 mog;
    cv::vector<Blob> blobContainer;
    unsigned int ID;
    /**************** Tracking ****************/
};

#endif // PLAYER_H
