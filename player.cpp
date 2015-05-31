#include "player.h"

const bool TRACKING_CONST = 0;
const bool HEAT_MAP = 0;

Player::Player(QObject *parent)
 : QThread(parent)
{
    stop = true;
    frameFeatures = new FrameFeatures(this);
    QThread::connect(frameFeatures, SIGNAL(onFeaturesFound(bool)), this, SLOT(onFeaturesPassed(bool)));
    QThread::connect(frameFeatures, SIGNAL(onDictionaryMade(bool)), this, SLOT(onDictionaryPassed(bool)));
}

Player::~Player()
{
    mutex.lock();
    stop = true;
    capture->release();
    delete capture;
    condition.wakeOne();
    mutex.unlock();
    wait();
}

bool Player::loadVideo(string filename) {
    capture  =  new cv::VideoCapture(filename);

    if (capture->isOpened())
    {
        frameRate = (int) capture->get(CV_CAP_PROP_FPS);
        ID = 0;
        frameFeatures->setFilename(filename);
        frameFeatures->start();
        return true;
    }
    else
        return false;
}

void Player::Play()
{
    if (!isRunning()) {
        if (isStopped()){
            stop = false;
        }
        start(LowPriority);
    }
}

void Player::Stop()
{
    stop = true;
}

void Player::msleep(int ms){
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
}

bool Player::isStopped() const{
    return this->stop;
}

double Player::getCurrentFrame()
{
    return capture->get(CV_CAP_PROP_POS_FRAMES);
}

double Player::getNumberOfFrames()
{
    return capture->get(CV_CAP_PROP_FRAME_COUNT);
}

double Player::getFrameRate()
{
    return frameRate;
}

void Player::setCurrentFrame( int frameNumber )
{
    capture->set(CV_CAP_PROP_POS_FRAMES, frameNumber);
}

void Player::run()
{
    int delay = (1000/frameRate);
    while(!stop)
    {
        if (!capture->read(frame))
        {
            stop = true;
        }
        if (frame.channels()== 3)
        {
            RGBframe = frame.clone();
            cv::cvtColor(frame, frame, CV_BGR2RGB);

            if(TRACKING_CONST)
                Tracking();

            img = QImage((const unsigned char*)(frame.data), frame.cols,frame.rows, QImage::Format_RGB888);
            imgProcessed = QImage((const unsigned char*)(processedFrame.data), processedFrame.cols,processedFrame.rows,QImage::Format_RGB888);

            emit originalImage(img);
            emit processedImage(imgProcessed);
        }
        else
        {
            RGBframe = frame.clone();
            cv::cvtColor(frame, frame, CV_BGR2RGB);

            if(TRACKING_CONST)
                Tracking();

            img = QImage((const unsigned char*)(frame.data), frame.cols,frame.rows,QImage::Format_Indexed8);
            imgProcessed = QImage((const unsigned char*)(processedFrame.data), processedFrame.cols,processedFrame.rows,QImage::Format_Indexed8);

            emit originalImage(img);
            emit processedImage(imgProcessed);
        }

        this->msleep(delay);
    }
}

void Player::ProcessFrame()
{

}

void Player::onFeaturesPassed(bool found)
{
    qDebug() << "features have been found: " << found;
    featureVectorPerFrame = frameFeatures->getFeatureVectors();

    if (found && HEAT_MAP   )
        getFeatureHeatMap();
}

void Player::onDictionaryPassed(bool voc)
{
    qDebug() << "dictionary has been created: " << voc;
    this->dictionary = frameFeatures->dictionary;
    emit dictionaryPassed(voc);
}

void Player::getFeatureHeatMap()
{
    vector<int> bins(featureVectorPerFrame.size());
    Mat heatMap(frame.rows, frame.cols, CV_8UC3, Scalar( 0,0,0));
    normalize(heatMap, heatMap, 0, heatMap.rows, NORM_MINMAX, -1, Mat() );
    Point2f point;
    vector<KeyPoint> framePoints;

    for(unsigned int i = 0; i < featureVectorPerFrame.size(); i++)
    {
        framePoints = featureVectorPerFrame.at(i);
        for(unsigned int j = 0; j < framePoints.size(); j++)
        {
            point = framePoints.at(j).pt;

            heatMap.at<int>(point.x, point.y) = 255;
        }

        //bins.at(i) = featureVectorPerFrame.at(i).size();
        //drawing.at(i, bins.at(i)) = 1;
        //qDebug() << "Size of bin " << i <<  "= " << bins.at(i);
    }

    cv::imshow("SURF Feature Heat Map", heatMap);
}

/**************** Tracking ****************/
bool sortByFrameCount(const Blob &lhs, const Blob &rhs) { return lhs.frameCount > rhs.frameCount; } // Sorting functions

void
Player::Tracking()
{
    int frame_number = getCurrentFrame();
    cv::vector<cv::vector<cv::Point> > contours; // Contour variables
    cv::vector<cv::Vec4i> hierarchy;
    rawFrame = frame.clone();
    rawCopyFrame = rawFrame.clone();

    mog(rawFrame, foregroundFrame, -1); // Background subtraction
    mog.set("nmixtures", 2);
    mog.set("detectShadows",0);

    // Threshold and morphology operations
    cv::threshold(foregroundFrame,foregroundFrame, 130, 255, cv::THRESH_BINARY);
    cv::medianBlur(foregroundFrame,foregroundFrame,3);
    cv::erode(foregroundFrame,foregroundFrame,cv::Mat());
    cv::dilate(foregroundFrame,foregroundFrame,cv::Mat());

    // Get foreground buffer
    foregroundFrameBuffer = foregroundFrame.clone();

    cv::findContours( foregroundFrame, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    // Initialize bounding rectangle for contours
    cv::Rect boundingRectangle;

    // Find Optical Flow features
    //vector<cv::Point> featuresCurrent;
    //goodFeaturesToTrack(foregroundFrameBuffer, featuresCurrent, 30, 0.01, 30); //calculate the features for use in next iteration

    // Object buffer
    vector<int> contourTaken(contours.size(),0);

    // Remove old objects with few frames
    for (vector<Blob>::iterator it = blobContainer.begin(); it!=blobContainer.end();) {
        //if the object hasnt appeared in the past 50 frames, or has only been present for at least 20 frames
        if (frame_number - it->lastFrameNumber > 50 && it->frames.size() < 20) {
            it = blobContainer.erase(it);
        } else {
            ++it;
        }
    }

    // Detect collisions and append object to object containers
    for (unsigned int bli = 0; bli < blobContainer.size(); bli++) {
        // Clean contact contours
        blobContainer[bli].contactContours.clear();
        blobContainer[bli].collision = 0;

        // Loop contours
        for( unsigned int coi = 0; coi<contours.size(); coi++ ) {
            // Obtain ROI from bounding rectangle of contours
            boundingRectangle = cv::boundingRect(contours[coi]);

            // Get distance
            float distance = sqrt(pow((blobContainer[bli].lastRectangle.x + blobContainer[bli].lastRectangle.width/2.0)-(boundingRectangle.x + boundingRectangle.width/2.0),2.0)+
            pow((blobContainer[bli].lastRectangle.y + blobContainer[bli].lastRectangle.height/2.0)-(boundingRectangle.y + boundingRectangle.height/2.0),2.0));

            // Detect collisions
            if (distance < min(blobContainer[bli].lastRectangle.width,blobContainer[bli].lastRectangle.height) &&//(distance < fmaxf(boundingRectangle.width,boundingRectangle.height)*2.0 &&
            frame_number - blobContainer[bli].lastFrameNumber == 1 &&
            max(boundingRectangle.width,boundingRectangle.height) > max(capture->get(CV_CAP_PROP_FRAME_WIDTH),capture->get(CV_CAP_PROP_FRAME_HEIGHT))/30)
            {
                blobContainer[bli].contactContours.push_back(coi);
            }
        }
    }

    // Sort blobContainer
    if (blobContainer.size() > 1)
        sort(blobContainer.begin(),blobContainer.end(), sortByFrameCount);

    // Find collision contours with biggest area
    for (unsigned int bli = 0; bli<blobContainer.size(); bli++) {
        unsigned int maxArea = 0;
        int selectedContourIndex = -1;
        for (unsigned int cni = 0; cni<blobContainer[bli].contactContours.size(); cni++) {
            int coi = blobContainer[bli].contactContours[cni];
            int contourArea = cv::boundingRect(contours[coi]).width*cv::boundingRect(contours[coi]).height;
            if (contourArea > maxArea) {
                maxArea = contourArea;
                selectedContourIndex = coi;
            }
        }

        //blobContainer[bli].contactContours.clear();

        // Append blob with largest area
        if (selectedContourIndex != -1 && contourTaken[selectedContourIndex] == 0) {
            contourTaken[selectedContourIndex] = 1;
            blobContainer[bli].contactContours.push_back(selectedContourIndex);

            // Get contour properties
            boundingRectangle = cv::boundingRect(contours[selectedContourIndex]);
            roiFrameMask =  foregroundFrameBuffer(boundingRectangle).clone();
            roiFrame = foregroundFrameBuffer(boundingRectangle).clone();
            roiFrameBuffer = rawFrame(boundingRectangle).clone();
            roiFrameBuffer.copyTo(roiFrame, roiFrameMask);

            // Append objects
            blobContainer[bli].frameCount++;
            blobContainer[bli].lastFrameNumber = frame_number;
            blobContainer[bli].lastRectangle = boundingRectangle;
            blobContainer[bli].frames.push_back(roiFrameBuffer);
            blobContainer[bli].avgWidth = .8*blobContainer[bli].avgWidth + .2*roiFrame.size().width;
            blobContainer[bli].avgHeight = .8*blobContainer[bli].avgHeight + .2*roiFrame.size().height;
            blobContainer[bli].maxWidth = max(blobContainer[bli].maxWidth,roiFrame.size().width);
            blobContainer[bli].maxHeight = max(blobContainer[bli].maxHeight,roiFrame.size().height);
        }

        if (blobContainer[bli].contactContours.size() > 1) {
            blobContainer[bli].collision = 1;
        }

    }

    // Create objects for the rest of the contours
    for (unsigned int coi = 0; coi < contours.size(); coi++) {
        boundingRectangle = cv::boundingRect(contours[coi]);
        if (contourTaken[coi] == 0 &&
        max(boundingRectangle.width,boundingRectangle.height) > max(capture->get(CV_CAP_PROP_FRAME_WIDTH),capture->get(CV_CAP_PROP_FRAME_HEIGHT))/20) {
            // Get contour properties
            boundingRectangle = cv::boundingRect(contours[coi]);
            roiFrameMask =  foregroundFrameBuffer(boundingRectangle).clone();
            roiFrame = foregroundFrameBuffer(boundingRectangle).clone();
            roiFrameBuffer = rawFrame(boundingRectangle).clone();
            roiFrameBuffer.copyTo(roiFrame, roiFrameMask);

            // Create objects
            ID++;
            Blob newObject;
            newObject.ID = ID;
            newObject.frameCount = 1;
            newObject.firstFrameNumber = frame_number;
            newObject.lastFrameNumber = frame_number;
            newObject.firstRectangle = boundingRectangle;
            newObject.lastRectangle = boundingRectangle;
            newObject.frames.push_back(roiFrameBuffer);
            newObject.avgWidth = roiFrame.size().width;
            newObject.avgHeight = roiFrame.size().height;
            newObject.maxWidth = roiFrame.size().width;
            newObject.maxHeight = roiFrame.size().height;
            //newObject.opticalFlowFeatures = featuresCurrent;
            blobContainer.push_back(newObject);

            qDebug() << "Object Created: " << ID;
        }
    }

    // Draw rectangles
    for (unsigned int bli = 0; bli<blobContainer.size(); bli++) {
        if (blobContainer[bli].lastFrameNumber == frame_number && blobContainer[bli].frameCount > 20) {//blobContainer[bli].frameCount > 10) {

            //rectangle(rawFrame, blobContainer[bli].lastRectangle, getRandomColorRGB(blobContainer[bli].ID));
            if (blobContainer[bli].collision == 1)
                rectangle(rawFrame, blobContainer[bli].lastRectangle, cv::Scalar(0,0,255),2);
            else
                rectangle(rawFrame, blobContainer[bli].lastRectangle, cv::Scalar(255,0,0),2);
        }
    }

    processedFrame = rawFrame.clone();
}

/**************** Tracking ****************/

