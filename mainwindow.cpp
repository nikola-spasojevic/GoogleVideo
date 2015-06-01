#include "mainwindow.h"
#include "ui_mainwindow.h"

const bool processed = false;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    bowDE(extractor, matcher)
{
    /**************** FRAME PROCESSING ****************/
    myPlayer = new Player();
    QObject::connect(myPlayer, SIGNAL(originalImage(QImage)), this, SLOT(updatePlayerUI(QImage)));
    QObject::connect(myPlayer, SIGNAL(processedImage(QImage)), this, SLOT(processedPlayerUI(QImage)));
    QObject::connect(myPlayer, SIGNAL(dictionaryPassed(bool)), this, SLOT(dictionaryReceived(bool)));
    /**************** FRAME PROCESSING ****************/

    ui->setupUi(this);
    ui->PlyBtn->setEnabled(false);
    ui->ffwdBtn->setEnabled(false);
    ui->horizontalSlider->setEnabled(false);

    /**************** FEATURE SELECTION ****************/
    window = cv::Rect(0, 0, 60, 60);//Window initialisation, setting width = 60, height = 60
    prevPoint = QPoint(0,0);
    topLeftCorner = QPoint(1600 ,1600);
    bottomRightCorner = QPoint(0,0);
    /**************** FEATURE SELECTION ****************/

    /**************** FEATURE DESCRIPTION AND EXTRACTION ****************/
    isDictionarySet = false;
    frame_counter = 0;
    feature_counter = 0;
    /**************** FEATURE DESCRIPTION AND EXTRACTION ****************/

    /**************** MOUSE TRACKING ****************/
    QWidget::connect(ui->outLabel, SIGNAL(Mouse_Move()), this, SLOT(Mouse_current_pos()));
    QWidget::connect(ui->outLabel, SIGNAL(Mouse_Pressed()), this, SLOT(Mouse_pressed()));
    QWidget::connect(ui->outLabel, SIGNAL(Mouse_Left()), this, SLOT(Mouse_left()));
    QWidget::connect(ui->outLabel, SIGNAL(Mouse_Release()), this, SLOT(Mouse_released()));
    /**************** MOUSE TRACKING ****************/
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::updatePlayerUI(QImage img)
{
    if(!processed)
    {
        if (!img.isNull())
        {
            px = QPixmap::fromImage(img);
            ui->outLabel->setScaledContents(true);
            ui->outLabel->setAlignment(Qt::AlignCenter);
            px = px.scaled(ui->outLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation);
            ui->outLabel->setPixmap(px);

            pxBuffer = px;
            ui->horizontalSlider->setValue(myPlayer->getCurrentFrame());
            ui->startTime->setText( getFormattedTime( (int)myPlayer->getCurrentFrame()/(int)myPlayer->getFrameRate()) );
        }
    }
}

void MainWindow::processedPlayerUI(QImage processedImg)
{
    if(processed)
    {
        if (!processedImg.isNull())
        {
            ui->outLabel->setScaledContents(true);
            ui->outLabel->setAlignment(Qt::AlignCenter);
            ui->outLabel->setPixmap(QPixmap::fromImage(processedImg).scaled(ui->outLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation));

            ui->horizontalSlider->setValue(myPlayer->getCurrentFrame());
            ui->startTime->setText( getFormattedTime( (int)myPlayer->getCurrentFrame()/(int)myPlayer->getFrameRate()) );
        }
    }
}

void MainWindow::dictionaryReceived(bool voc)
{
    isDictionarySet = true;
    qDebug() << "is dictionary empty: " << myPlayer->dictionary.empty();
    stop_idxs.clear(); //TODO: reinitaisle at correct place - when searching for new object!!!

    bowDE.setVocabulary(myPlayer->dictionary);
    vector< vector<KeyPoint> > featureVec = myPlayer->frameFeatures->keypoints_frameVector;
    vector<vector<float> > histogramsPerFrame(featureVec.size());
    vector<float> histogram_sum(DICTIONARY_SIZE, 0);
    vector<int> visualWord_count(DICTIONARY_SIZE,0); //number of occurences of frames containg a specific word (feature)
    std::vector<std::pair<double, int> > sum_idx_pairVector;

    for (int i = 0; i < featureVec.size(); i++)
    {
        cv::Mat frame = myPlayer->frameFeatures->frameVector.at(i);
        Mat histogram;
        bowDE.compute(frame, featureVec.at(i), histogram);

        for (int j = 0; j < DICTIONARY_SIZE; j++)
        {
            if(histogram.at<float>(0,j) > 0.)
            {
                visualWord_count.at(j)++;//create a word count for each visual word
            }
        }

        const float* p = histogram.ptr<float>(0);
        std::vector<float> vec(p, p + histogram.cols);
        for(int k = 0; k <DICTIONARY_SIZE; k++)
            histogram_sum.at(k) += vec.at(k);

        qDebug() << "Sum calculated!";

        histogramsPerFrame.at(i) = vec;
    }

    //Stop List - create  vector of indexes to be avoided when processing
    for (int i = 0; i < histogram_sum.size(); ++i)
    {
        sum_idx_pairVector.push_back(std::pair<float, int>(histogram_sum.at(i), i));
    }

    std::sort(sum_idx_pairVector.begin(), sum_idx_pairVector.end(), HelperFunctions::sort_pred());
    qDebug() << "Pairs sorted!";

    for (int j = 0; j < DICTIONARY_SIZE/20; j++) //remove top and bottom 5%
    {
        int temp_idx1 = sum_idx_pairVector.at(j).second;
        int temp_idx2 = sum_idx_pairVector.at(sum_idx_pairVector.size() - 1 - j).second;

        stop_idxs.push_back(temp_idx1);
        stop_idxs.push_back(temp_idx2);
    }
    qDebug() << "Stop indexes received!";

    for (int k = 0; k < stop_idxs.size(); k++)
        qDebug() << stop_idxs.at(k);

    //term fewquency - inverse document term added to word frequency - tf-idf
    for (int i = 0; i < featureVec.size(); i++)
    {
        vector<float> tf_idf;

        for (int j = 0; j < DICTIONARY_SIZE; j++)
        {
            int word_count = visualWord_count.at(j);
            qDebug() << "visual word count of " << j << " = " << word_count;

            if(std::find(stop_idxs.begin(), stop_idxs.end(), j) == stop_idxs.end())
            {
                if (word_count == 0)
                {
                    tf_idf.push_back(0.);
                }
                else
                {
                    float word_freq = histogramsPerFrame.at(i).at(j) * DICTIONARY_SIZE/featureVec.at(i).size();
                    qDebug() << "word frequency = " << word_freq;

                    float inverse_doc_freq = log(featureVec.size() / (float) word_count);
                    qDebug() << "inverse document frequency = " << inverse_doc_freq;

                    float weighted_freq = word_freq * inverse_doc_freq;
                    qDebug() << "tf-idf frequency component = " << weighted_freq;
                    tf_idf.push_back(weighted_freq);

                    //for word j, assign frame i as: the word is contained in the following documents
                    invIdxStruct[j].push_back(i);
                }

            }
        }

        tf_idfVectors.push_back(tf_idf);

        qDebug() << "size of a tf-idf container " << i << ": " <<  tf_idf.size();
    }



}

QString MainWindow::getFormattedTime(int timeInSeconds){

    int seconds = (int) (timeInSeconds) % 60 ;
    int minutes = (int) ((timeInSeconds / 60) % 60);
    int hours   = (int) ((timeInSeconds / (60*60)) % 24);

    QTime t(hours, minutes, seconds);
    if (hours == 0 )
        return t.toString("mm:ss");
    else
        return t.toString("h:mm:ss");
}

void MainWindow::on_PlyBtn_clicked()
{
    if (myPlayer->isStopped())
    {
        myPlayer->Play();
        ui->PlyBtn->setText(tr("Stop"));
    }else
    {
        myPlayer->Stop();
        ui->PlyBtn->setText(tr("Play"));
    }
}

void MainWindow::on_LdBtn_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Video"), ".", tr("Video Files (*.avi *.mpg *.mp4)"));
    QFileInfo name = filename;

    if (!filename.isEmpty())
    {
        if (!myPlayer->loadVideo(filename.toLatin1().data()))
        {
            QMessageBox msgBox;
            msgBox.setText("The selected video could not be opened!");
            msgBox.exec();
        }
        else
        {
           this->setWindowTitle(name.fileName());
           ui->PlyBtn->setEnabled(true);
           ui->ffwdBtn->setEnabled(true);
           ui->horizontalSlider->setEnabled(true);
           ui->horizontalSlider->setMaximum(myPlayer->getNumberOfFrames());
           ui->endTime->setText( getFormattedTime( (int)myPlayer->getNumberOfFrames()/(int)myPlayer->getFrameRate()) );
       }
    }
}

void MainWindow::on_horizontalSlider_sliderPressed()
{
    myPlayer->Stop();
}

void MainWindow::on_horizontalSlider_sliderReleased()
{
    myPlayer->Play();
}

void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    myPlayer->setCurrentFrame(position);
    ui->startTime->setText( getFormattedTime( position/(int)myPlayer->getFrameRate()) );
}

void MainWindow::on_ffwdBtn_pressed()
{ 
    const int FFValue = myPlayer->getFrameRate();
    int delay = (1000/FFValue);
    myPlayer->Stop();
    int j = myPlayer->getCurrentFrame();

    j += FFValue;
    myPlayer->setCurrentFrame(j);
    myPlayer->msleep(delay);
}

void MainWindow::on_ffwdBtn_released()
{
    myPlayer->Play();
}

void MainWindow::Mouse_current_pos()
{
    if (ui->outLabel->mouseHeld() )
    {
    if (frame_counter == FRAME_FREQ)
    {
        QPoint mouse_pos = ui->outLabel->mouseCurrentPos();

        if(mouse_pos.x() < ui->outLabel->width() && mouse_pos.y() < ui->outLabel->height() && mouse_pos.x() > 0 && mouse_pos.y() > 0 && prevPoint != QPoint(0,0))
        {   
            if (mouse_pos.x() > window.width)
                topLeftCorner.setX(mouse_pos.x() - window.width);
            else
                topLeftCorner.setX(0);

            if (mouse_pos.y() > window.height)
                topLeftCorner.setY(mouse_pos.y() - window.height);
            else
                topLeftCorner.setY(0);

            if (mouse_pos.x() < (ui->outLabel->width() - window.width))
                bottomRightCorner.setX(mouse_pos.x() + window.width);
            else
                bottomRightCorner.setX(ui->outLabel->width());

            if (mouse_pos.y() < (ui->outLabel->height() - window.height) )
                bottomRightCorner.setY(mouse_pos.y() + window.height);
            else
                bottomRightCorner.setY(ui->outLabel->height());
        }

        if (mouse_pos.x() < ui->outLabel->width() && mouse_pos.y() < ui->outLabel->height() && mouse_pos.x() > 0 && mouse_pos.y() > 0 && prevPoint != QPoint(0,0))
        {
            Rect roi_rect = Rect(topLeftCorner.x(), topLeftCorner.y(), bottomRightCorner.x() - topLeftCorner.x(), bottomRightCorner.y() - topLeftCorner.y());
            //Mat foreground = foregroundExtraction(roi_rect);
            Mat foreground = connectedComponents(roi_rect);
            processROI(foreground);
        }

        prevPoint = mouse_pos;
        frame_counter = 0; //process every frame according to frame frequency
    }

    frame_counter++;
    qDebug() << "frame counter: " << frame_counter;
    }
}

void MainWindow::Mouse_pressed()
{
    prevPoint = QPoint(0,0);
    keypoints_object.clear();
}

void MainWindow::Mouse_released()
{
    if (isDictionarySet && feature_counter > MIN_FEATURE_SIZE)
    {
        vector<vector<float> > histogramsPerRoi(roiVector.size());
        vector<int> visualWord_count(DICTIONARY_SIZE,0); //number of occurences of frames containg a specific word (feature)
        vector<float> tfROI(tf_idfVectors.at(0).size(), 0.);

        qDebug() << "Size of assumed ROI tf-idf vector: " << tfROI.size();

        for (int i = 0; i < roiVector.size(); i++)
        {
            Mat histogram;
            cv::Mat roi = roiVector.at(i);
            bowDE.compute(roi, keypointVector.at(i), histogram);

            for (int j = 0; j < DICTIONARY_SIZE; j++)
            {
                if(histogram.at<float>(0,j) > 0.)
                {
                    visualWord_count.at(j)++;//create a word count for each visual word
                }
            }

            const float* p = histogram.ptr<float>(0);
            std::vector<float> vec(p, p + histogram.cols); 
            histogramsPerRoi.at(i) = vec;
        }

        //term frequency - inverse document term added to word frequency - tf-idf
        for (int i = 0; i < roiVector.size(); i++)
        {
            vector<float> tf_idf;

            for (int j = 0; j < DICTIONARY_SIZE; j++)
            {
                int word_count = visualWord_count.at(j);
                qDebug() << "visual word count = " << word_count;

                if(std::find(stop_idxs.begin(), stop_idxs.end(), j) == stop_idxs.end())
                {
                    if (word_count == 0)
                    {
                        tf_idf.push_back(0.);
                    }
                    else
                    {
                        float word_freq = histogramsPerRoi.at(i).at(j) * DICTIONARY_SIZE/keypointVector.at(i).size();;
                        qDebug() << "word frequency = " << word_freq;

                        float inverse_doc_freq = log(roiVector.size() / (float) word_count);
                        qDebug() << "inverse document frequency = " << inverse_doc_freq;

                        float weighted_freq = word_freq * inverse_doc_freq;
                        qDebug() << "tf-idf frequency component = " << weighted_freq;
                        tf_idf.push_back(weighted_freq);
                    }
                }
            }

            qDebug() << "Size of temporary ROI tf-idf vector: " << tf_idf.size();

            for(int k = 0; k < tfROI.size(); k++)
            {
                tfROI.at(k) =+ tf_idf.at(k); //accumulate all of the wighted frequncies of all visaul words occuring in all of the ROI querries
                qDebug() << tf_idf.at(k);
            }
        }

        //Should check how often certain words appear and average them accroding to that, rather than the size of the entire set
        qDebug() << "Averaging...";
        for(int k = 0; k < tfROI.size(); k++) // the inverted file index structure has 10% less members than DICTIONARY_SIZE
        {
            tfROI.at(k) /= roiVector.size(); //by averaging the weighted frequncies , we supress the noise whilst maintaining the dominant words
        }

        qDebug() << "Averaged!!!";

        querryFrames(tfROI);
    }
}

void MainWindow::Mouse_left()
{
    ui->outLabel->left = true;
}


Mat MainWindow::connectedComponents(Rect roi_rect)
{
    pxBuffer = pxBuffer.scaled(ui->outLabel->size());
    cv::Mat frame = HelperFunctions::QPixmapToCvMat(pxBuffer);
    cv::Mat roi = frame(roi_rect);
    cv::Mat roiBuffer;
    cv::Mat roiBufferCopy = roi.clone();

   ///-- find dominant object via contours and calculate surf feature points within the bounded region--//
    cv::cvtColor(roi, roiBuffer, CV_BGR2GRAY);

    // Threshold and morphology operations
    adaptiveThreshold(roiBuffer, roiBuffer, 255, ADAPTIVE_THRESH_GAUSSIAN_C,  THRESH_BINARY, 5, 0);
    cv::medianBlur(roiBuffer,roiBuffer,3);
    //equalizeHist(roiBuffer, roiBuffer);
    cv::erode(roiBuffer,roiBuffer,cv::Mat());
    cv::dilate(roiBuffer,roiBuffer,cv::Mat());
    GaussianBlur(roiBuffer, roiBuffer, Size(7,7), 1.5, 1.5);    

    cv::vector<cv::vector<cv::Point> > contours;
    findContours( roiBuffer, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    qDebug() << contours.size();

    /// Find contour with largest area
    double maxArea = 0;
    int maxIdx;
    for( int i = 0; i < contours.size(); i++ )
    {
        double ctrArea = contourArea(contours[i]);
        if ( maxArea < ctrArea)
        {
            maxArea = ctrArea;
            maxIdx = i;
        }
    }
    if (roiBuffer.size().area() == contourArea(contours[maxIdx]) )
        qDebug() << "Contour exceeds bounds!!!";
    qDebug() << maxArea;

    if (maxArea > COUNTOUR_AREA_THRESHOLD)
    {
        window.width *= 0.8;
        window.height*= 0.8;
    }
    else
    {
        window.width *= 1.2;
        window.height*= 1.2;
    }

    qDebug() << "window: " << window.width << " x " << window.height;

    Mat contourROI;
    Mat mask = Mat::zeros( roi.size(), roi.type());
    drawContours( mask, contours, maxIdx, Scalar(255,255,255), CV_FILLED, 8);

    drawContours( roiBufferCopy, contours, maxIdx, Scalar(0,150,0), 1.5 , 8);
    cv::Rect roi_temp(Point(topLeftCorner.x(), topLeftCorner.y()), roi.size());

    //Pixmap to Mat
    pxBuffer = pxBuffer.scaled(ui->outLabel->size());
    roiBufferCopy.copyTo(frame(roi_temp));
    px = HelperFunctions::cvMatToQPixmap(frame);
    ui->outLabel->setScaledContents(true);
    ui->outLabel->setAlignment(Qt::AlignCenter);
    px = px.scaled(ui->outLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation);

    QPainter p(&px);
    QPen pen(Qt::red);
    pen.setWidth( 2 );
    p.setPen(pen);
    QPoint mouse_pos = ui->outLabel->mouseCurrentPos();
    p.drawLine (mouse_pos.x(), mouse_pos.y(), prevPoint.x(), prevPoint.y());
    p.end();

    ui->outLabel->setPixmap(px);

    qDebug() << "frame set";

    bitwise_and(mask, roi, contourROI);
    return contourROI;
    ///-- find dominant object via contours and calculate surf feature points within the bounded region--//
}

Mat MainWindow::foregroundExtraction(Rect roi_rect)
{
    pxBuffer = pxBuffer.scaled(ui->outLabel->size());
    cv::Mat frame = HelperFunctions::QPixmapToCvMat(pxBuffer);
    Mat frame_buffer;
    cv::cvtColor(frame, frame_buffer, CV_RGBA2RGB);

    Mat mask, bgdModel, fgdModel;

    grabCut(frame_buffer,   //input image
            mask,           //segmentation result
            roi_rect,       //rectangle containing foreground
            bgdModel, fgdModel, //models
            1,              //number of iterations
            GC_INIT_WITH_RECT); //use rectangle

    // Get the pixels marked as likely foreground
    cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);

    // Generate output image
    cv::Mat foreground(frame_buffer.size(), frame.type(), cv::Scalar(255,255,255));
    frame.copyTo(foreground, mask); // backgrounnd pixels not copied
    Mat output = foreground(roi_rect);

    //cv::namedWindow("Selected Feature");
    //cv::imshow("Selected Feature", output);

    return output;
}

void MainWindow::querryFrames(vector<float> queryVec)
{
    vector<std::pair<int, float> > retrievedframeAglesMap;
    vector<int> passedFrameIdxs;
    //Traverse through the frames and find ones with a distance beneath a certain thrshold
    //first, find thw non-zero words within the query vector
    for(int i = 0; i < queryVec.size(); i++)
    {
        if(queryVec.at(i) != 0) // figure out how to avoid close to zero values
        {
            qDebug() << "index number: " << i;
            vector<int> qf = invIdxStruct[i];//querrying the inverse index structure, checking which frames are associated with the querried visual word

            for(std::vector<int>::iterator it = qf.begin(); it != qf.end(); it++)//the iterator is a frame index value, find which frames contain this specific visual word
            {
                if(std::find(passedFrameIdxs.begin(), passedFrameIdxs.end(), *it) == passedFrameIdxs.end())//make sure you havent already traversed through that frame before
                {
                    std::pair<int, float> frameIdxAngle; //pair of cosine angle value and frame index
                    float fd = 0, norm_q = 0, norm_d = 0;
                    vector<float> docVec = tf_idfVectors.at(*it);

                    //calculate the cosine distance between the querry vector and the frame vector
                    for(int c = 0; c < queryVec.size(); c++)
                    {
                        float q = queryVec.at(c); //querry value
                        float d = docVec.at(c);//document value

                        fd += q * d;
                        norm_q += q*q;
                        norm_d += d*d;
                    }

                    //normalise the cosine distance/frequency score
                    fd /= sqrt(norm_q);
                    fd /= sqrt(norm_d);

                    qDebug() << "Calculated pair: index -> " << *it << ", cosine distace/frequency score -> " << fd;

                    frameIdxAngle.first = *it;
                    frameIdxAngle.second = fd;

                    retrievedframeAglesMap.push_back(frameIdxAngle);
                    passedFrameIdxs.push_back(*it);
                }
            }
        }
    }

    sort(retrievedframeAglesMap.begin(), retrievedframeAglesMap.end(), HelperFunctions::sort_frameIdxPairs());

    for (int k = 0; k < retrievedframeAglesMap.size(); k++)
    {
        qDebug() << "Object appearing in frames: " << retrievedframeAglesMap.at(k).first << ", by distance: " << retrievedframeAglesMap.at(k).second;
    }

    SpatialConsistency(retrievedframeAglesMap);
}

void MainWindow::SpatialConsistency(vector<std::pair<int, float> > retrievedIdxAglesMap)
{
    //Rerank the top Ns retrieved keyframes using the spatioal consistency check

    for (int i = 0; i < retrievedIdxAglesMap.size(); i++)
    {
        std::pair<int, float> curr_pair = retrievedIdxAglesMap.at(i);
        int frameIdx =    curr_pair.first;
        int cosDistance = curr_pair.second;

        vector<KeyPoint> curr_KeypointVector = keypointVector.at(frameIdx);
        Mat curr_frame = myPlayer->frameFeatures->frameVector.at(frameIdx);

        //reranking the frames based on a measure of spatial consistency
        //choose 10 random keypoints and calculate how many points there are in a certain radius and check if they're match has a similair number

        //look how similar the k-nearest neighbours are!

        //first, we will traverse through the top 20 roi frames and compare each of they're sptail consistencies with the
        //look for SIFT correspondecens using homography and RANSAC
        for(int k = 0; k < )




    }
}

void MainWindow::processROI(Mat roi)
{ 
    vector<KeyPoint> keypoints_object;
    cvtColor(roi, roi, COLOR_RGBA2RGB);
    detector->detect(roi,keypoints_object);
    roiVector.push_back(roi);
    keypointVector.push_back(keypoints_object);
    feature_counter += keypoints_object.size();
}


