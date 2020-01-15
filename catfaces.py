import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
face_detector = cv.CascadeClassifier('haarcascade_frontalcatface.xml')
face_ext_detector = cv.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
face_lbp_detector = cv.CascadeClassifier('lbpcascade_frontalcatface.xml')
facemark = cv.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')

assert cap.isOpened()


while True:
    ret, frame = cap.read()
    faces1 = []
    faces2 = []
    faces3 = []

    if not ret:
        print('Err')
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray)
    if len(faces) != 0:
        faces[:,2:] += faces[:,:2]
        faces1 = faces
    faces = face_ext_detector.detectMultiScale(gray)
    if len(faces) != 0:
        faces[:,2:] += faces[:,:2]
        faces3 = faces
    faces = face_lbp_detector.detectMultiScale(gray)
    if len(faces) != 0:
        faces[:,2:] += faces[:,:2]
        faces3 = faces


    for face in faces1:
        x1, y1, x2, y2 = face
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))
    for face in faces2:
        x1, y1, x2, y2 = face
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0))
    for face in faces3:
        x1, y1, x2, y2 = face
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

    cv.flip(frame, 1, frame)

    cv.imshow('frame', frame)

    if cv.waitKey(5) >= 0:
        break

cap.release()
cv.destroyAllWindows()
