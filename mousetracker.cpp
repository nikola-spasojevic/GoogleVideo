#include "mousetracker.h"

MouseTracker::MouseTracker(QWidget *parent)
    : QLabel(parent)
{
    left = false;
    hold = false;
}

MouseTracker::~MouseTracker()
{
}

void MouseTracker::mouseMoveEvent(QMouseEvent *ev)
{
        this->x = ev->x();
        this->y = ev->y();
        emit Mouse_Move();
}

void MouseTracker::mousePressEvent(QMouseEvent *ev)
{
    if (ev->button() == Qt::LeftButton)
    {
        hold = true;
        this->beginPoint = ev->pos();
        emit Mouse_Pressed();
    }
}

void MouseTracker::mouseLeaveEvent(QEvent *)
{
    left = true;
    emit Mouse_Left();
}

bool MouseTracker::mouseLeft() const
{
    return this->left;
}

bool MouseTracker::mouseHeld() const
{
    return this->hold;
}

void MouseTracker::mouseReleaseEvent(QMouseEvent *ev)
{
    hold = false;
    this->endPoint = ev->pos();
    emit Mouse_Release();
}

QPoint MouseTracker::mouseCurrentPos()
{
    return QPoint(this->x, this->y);
}

