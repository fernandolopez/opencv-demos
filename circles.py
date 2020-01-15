import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
assert cap.isOpened()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    t = 130

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 2, 20, param1=t, param2=200, minRadius=10, maxRadius=200)

    bw = cv.Canny(frame, t / 2, t)
    if circles is not None:
        for x, y, r in circles[0]:
            cv.circle(frame, (x, y), r, (0, 255, 0), 3)

    bw = cv.cvtColor(bw, cv.COLOR_GRAY2BGR)
    cv.add(frame, bw, frame)

    frame = cv.flip(frame, 1)
    cv.imshow('frame', frame)

    if cv.waitKey(5) >= 0:
        break

cap.release()
cv.destroyAllWindows()
