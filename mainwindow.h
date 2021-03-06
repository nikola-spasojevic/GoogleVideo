#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <player.h>
#include <mousetracker.h>
#include <helperfunctions.h>
#include <QtGui>
#include <opencv2/ml/ml.hpp>

#define DICTIONARY_SIZE 1500
#define MIN_FEATURE_SIZE 2000
#define COUNTOUR_AREA_THRESHOLD 3000
#define FRAME_FREQ 3

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    Player* myPlayer;


    vector<vector<float> > tf_idfVectors;
    QPoint prevPoint;
    QPoint topLeftCorner;
    QPoint bottomRightCorner;
    QPixmap px, pxBuffer;
    cv::Rect window;
    //Ptr<FeatureDetector> detector = new PyramidAdaptedFeatureDetector(new DynamicAdaptedFeatureDetector (new SurfAdjuster(700,true), 100, 200, 2), 4);
    //Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
    //Ptr<FeatureDetector> detector = new DynamicAdaptedFeatureDetector ( new SurfAdjuster(700,true), 10, 500, 10);
    Ptr<FeatureDetector> detector = new cv::MserFeatureDetector();
    Ptr<DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bowDE;
    bool isDictionarySet;
    vector<KeyPoint> keypoints_object;
    int frame_counter;
    vector<vector<KeyPoint> > keypointVector;
    vector<Mat> roiVector;
    int feature_counter;
    vector<int> invIdxStruct[DICTIONARY_SIZE]; //word indexes corresponding to frame indexes
    std::vector<int> stop_idxs; //indexes of visual words corresponding to the histograms we need to avoid

    void processROI(Mat roi);
    Mat connectedComponents(Rect roi_rect);
    Mat foregroundExtraction(Rect roi_rect);
    void querryFrames(vector<float> tfROI);
    void retrievalStage(vector<float> querryVec);
    void SpatialConsistency(vector<std::pair<int, float> > retrievedframeAglesMap);


signals:

private slots:
    /**************** FRAME PROCESSING ****************/
    void updatePlayerUI(QImage img);
    void processedPlayerUI(QImage processedImg);
    void dictionaryReceived(bool voc);
    /**************** FRAME PROCESSING ****************/

    /**************** PLAYER ****************/
    void on_LdBtn_clicked();
    void on_PlyBtn_clicked();
    QString getFormattedTime(int timeInSeconds);
    void on_horizontalSlider_sliderPressed();
    void on_horizontalSlider_sliderReleased();
    void on_horizontalSlider_sliderMoved(int position);
    void on_ffwdBtn_pressed();
    void on_ffwdBtn_released();
    /**************** PLAYER ****************/

    /**************** MOUSE TRACKER ****************/
    void Mouse_current_pos();
    void Mouse_pressed();
    void Mouse_left();
    void Mouse_released();
    /**************** MOUSE TRACKER ****************/
};

#endif // MAINWINDOW_H
