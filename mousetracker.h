#ifndef MOUSETRACKER_H
#define MOUSETRACKER_H

#include <QMouseEvent>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDebug>
#include <QEvent>
#include <QLabel>
#include <QPoint>

class MouseTracker : public QLabel
{
    Q_OBJECT

private:

protected:

public:
    explicit MouseTracker(QWidget *parent = 0);
    ~MouseTracker();

    void mouseMoveEvent(QMouseEvent *ev);
    void mousePressEvent(QMouseEvent *ev);
    void mouseLeaveEvent(QEvent *);
    void mouseReleaseEvent(QMouseEvent *ev);
    QPoint mouseCurrentPos();

    bool left, hold;
    int x, y;

    QPoint beginPoint;
    QPoint endPoint;

    bool mouseLeft() const;
    bool mouseHeld() const;

signals:
    void Mouse_Pressed();
    void Mouse_Move();
    void Mouse_Left();
    void Mouse_Hold();
    void Mouse_Release();
};
#endif // MOUSETRACKER_H
