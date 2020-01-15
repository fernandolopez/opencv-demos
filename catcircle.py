import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
face_detector = cv.CascadeClassifier('haarcascade_frontalcatface.xml')

assert cap.isOpened()


while True:
    ret, frame = cap.read()

    if not ret:
        print('Err')
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray)
    if len(faces) != 0:
        faces[:,2:] += faces[:,:2]
    else:
        frame = np.zeros_like(frame)

    for face in faces:
        x1, y1, x2, y2 = face
        dx = abs(x2 - x1) / 2
        dy = abs(y2 - y1) / 2
        r = int(max(dx, dy))
        x = int(dx + min(x1, x2))
        y = int(dy + min(y1, y2))
        black = np.zeros_like(frame)
        cv.circle(black, (x, y), r, (255, 255, 255), cv.FILLED)
        frame = cv.bitwise_and(frame, black)

    cv.flip(frame, 1, frame)

    cv.imshow('frame', frame)

    if cv.waitKey(5) >= 0:
        break

cap.release()
cv.destroyAllWindows()
