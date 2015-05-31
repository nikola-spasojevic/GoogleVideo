#-------------------------------------------------
#
# Project created by QtCreator 2015-04-16T21:01:42
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ClassificationPlayer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    player.cpp \
    framefeatures.cpp \
    mousetracker.cpp

HEADERS  += mainwindow.h \
    player.h \
    framefeatures.h \
    blob.h \
    mousetracker.h \
    helperfunctions.h

FORMS    += mainwindow.ui

LIBS += -L/usr/local/Cellar/opencv/2.4.9/lib \
-lopencv_core \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_flann \
-lopencv_legacy \
-lopencv_ml \
-lopencv_features2d \
-lopencv_calib3d \
-lopencv_nonfree \
-lopencv_video \
-lopencv_objdetect

INCLUDEPATH += $$PWD/../../../../../usr/local/Cellar/opencv/2.4.9/include
DEPENDPATH += $$PWD/../../../../../usr/local/Cellar/opencv/2.4.9/include

DISTFILES +=
