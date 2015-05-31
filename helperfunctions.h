#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

/*
   Functions to convert between OpenCV's cv::Mat and Qt's QImage and QPixmap.

   Andy Maloney
   23 November 2013
   http://asmaloney.com/2013/11/code/converting-between-cvmat-and-qimage-or-qpixmap
 */

#include <QImage>
#include <QPixmap>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"


namespace HelperFunctions {
   inline QImage  cvMatToQImage( const cv::Mat &inMat )
   {
      switch ( inMat.type() )
      {
         // 8-bit, 4 channel
         case CV_8UC4:
         {
            QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB32 );

            return image;
         }

         // 8-bit, 3 channel
         case CV_8UC3:
         {
            QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB888 );

            return image.rgbSwapped();
         }

         // 8-bit, 1 channel
         case CV_8UC1:
         {
            static QVector<QRgb>  sColorTable;

            // only create our color table once
            if ( sColorTable.isEmpty() )
            {
               for ( int i = 0; i < 256; ++i )
                  sColorTable.push_back( qRgb( i, i, i ) );
            }

            QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_Indexed8 );

            image.setColorTable( sColorTable );

            return image;
         }

         default:
            qWarning() << "ASM::cvMatToQImage() - cv::Mat image type not handled in switch:" << inMat.type();
            break;
      }

      return QImage();
   }

   inline QPixmap cvMatToQPixmap( const cv::Mat &inMat )
   {
      return QPixmap::fromImage( cvMatToQImage( inMat ) );
   }

   // If inImage exists for the lifetime of the resulting cv::Mat, pass false to inCloneImageData to share inImage's
    // data with the cv::Mat directly
    //    NOTE: Format_RGB888 is an exception since we need to use a local QImage and thus must clone the data regardless
    inline cv::Mat QImageToCvMat( const QImage &inImage, bool inCloneImageData = true )
    {
       switch ( inImage.format() )
       {
          // 8-bit, 4 channel
          case QImage::Format_RGB32:
          {
             cv::Mat  mat( inImage.height(), inImage.width(), CV_8UC4, const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine() );

             return (inCloneImageData ? mat.clone() : mat);
          }

          // 8-bit, 3 channel
          case QImage::Format_RGB888:
          {
             if ( !inCloneImageData )
                qWarning() << "ASM::QImageToCvMat() - Conversion requires cloning since we use a temporary QImage";

             QImage   swapped = inImage.rgbSwapped();

             return cv::Mat( swapped.height(), swapped.width(), CV_8UC3, const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine() ).clone();
          }

          // 8-bit, 1 channel
          case QImage::Format_Indexed8:
          {
             cv::Mat  mat( inImage.height(), inImage.width(), CV_8UC1, const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine() );

             return (inCloneImageData ? mat.clone() : mat);
          }

          default:
             qWarning() << "ASM::QImageToCvMat() - QImage format not handled in switch:" << inImage.format();
             break;
       }

       return cv::Mat();
    }

    // If inPixmap exists for the lifetime of the resulting cv::Mat, pass false to inCloneImageData to share inPixmap's data
    // with the cv::Mat directly
    //    NOTE: Format_RGB888 is an exception since we need to use a local QImage and thus must clone the data regardless
    inline cv::Mat QPixmapToCvMat( const QPixmap &inPixmap, bool inCloneImageData = true )
    {
       return QImageToCvMat( inPixmap.toImage(), inCloneImageData );
    }

    inline bool niceHomography(const cv::Mat* H)
    {
      const double det = H->at<double>(0,0) * H->at<double>(1,1) - H->at<double>(1,0) * H->at<double>(0,1);

      if (det < 0)
        return false;

      const double N1 = sqrt(H->at<double>(0,0) * H->at<double>(0,0) + H->at<double>(1,0) * H->at<double>(1,0));
      if (N1 > 4 || N1 < 0.1)
        return false;

      const double N2 = sqrt(H->at<double>(0,1) * H->at<double>(0,1) + H->at<double>(1,1) * H->at<double>(1,1));
      if (N2 > 4 || N2 < 0.1)
        return false;

      const double N3 = sqrt(H->at<double>(2,0) * H->at<double>(2,0) + H->at<double>(2,1) * H->at<double>(2,1));
      if (N3 > 0.002)
        return false;

      return true;
    }

    inline void cleanPreviousWindows()
    {
        waitKey(1);
        destroyAllWindows();
        waitKey(1);
    }

    inline cv::Mat createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size)
    {
        // let's first find out the maximum dimensions
        int max_width = 0;
        int max_height = 0;
        for ( int i = 0; i < images.size(); i++) {
            // check if type is correct
            // you could actually remove that check and convert the image
            // in question to a specific type
            if ( i > 0 && images[i].type() != images[i-1].type() ) {
                qDebug()  << "WARNING:createOne failed, different types of images";
                return cv::Mat();
            }
            max_height = std::max(max_height, images[i].rows);
            max_width = std::max(max_width, images[i].cols);
        }
        // number of images in y direction
        int rows = std::ceil(images.size() / cols);

        // create our result-matrix
        cv::Mat result = cv::Mat::zeros(rows*max_height + (rows-1)*min_gap_size,
                                        cols*max_width + (cols-1)*min_gap_size, images[0].type());
        size_t i = 0;
        int current_height = 0;
        int current_width = 0;
        for ( int y = 0; y < rows; y++ ) {
            for ( int x = 0; x < cols; x++ ) {
                if ( i >= images.size() ) // shouldn't happen, but let's be safe
                    return result;
                // get the ROI in our result-image
                cv::Mat to(result,
                           cv::Range(current_height, current_height + images[i].rows),
                           cv::Range(current_width, current_width + images[i].cols));
                // copy the current image to the ROI
                images[i++].copyTo(to);
                current_width += max_width + min_gap_size;
            }
            // next line - reset width and update height
            current_width = 0;
            current_height += max_height + min_gap_size;
        }
        return result;
    }

    inline double normalise(double dist, double min_dist, double max_dist)
    {
        return (dist - min_dist)/(max_dist - min_dist);
    }

    struct sort_pred {
        bool operator()(const std::pair<float,int> &left, const std::pair<float,int> &right) {
            return left.first < right.first;
        }
    };

    struct sort_frameIdxPairs {
        bool operator()(const std::pair<int,float> &left, const std::pair<int,float> &right) {
            return left.second < right.second;
        }
    };

 }
#endif // HELPERFUNCTIONS_H
