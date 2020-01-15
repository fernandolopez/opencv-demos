import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
assert cap.isOpened()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv.blur(frame, (10, 10))
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (19, 86, 6), (64, 255, 255))
    mask = cv.erode(mask, None, iterations=3)
    mask = cv.dilate(mask, None, iterations=3)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    if len(contours) > 0:
        c = max(contours, key=cv.contourArea)
        ((x, y), r) = cv.minEnclosingCircle(c)
        if r > 20:
            x = int(x)
            y = int(y)
            r = int(r)

            black = np.zeros_like(frame)
            cv.circle(black, (x, y), r, (255, 255, 255), cv.FILLED)
            frame = cv.bitwise_and(frame, black)

    cv.flip(frame, 1, frame)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)

    if cv.waitKey(5) >= 0:
        break

cap.release()
cv.destroyAllWindows()
