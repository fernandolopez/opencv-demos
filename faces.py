import cv2 as cv
import numpy as np

def draw_polyline(frame, landmarks, start, end, is_closed=False, color=(0, 255, 0)):
    mask = np.zeros_like(frame)
    circle = np.zeros_like(frame)

    points = landmarks[start:end+1]
    #cv.polylines(frame, np.int32([points]), is_closed, color, 2, 16)
    #cv.polylines(frame, np.int32([points]), is_closed, color, 2, 16)
    ellipse = cv.fitEllipse(points)
    cv.ellipse(mask, ellipse, (255, 255, 255), cv.FILLED)
    cv.ellipse(frame, ellipse, color, cv.FILLED)

    x, y = map(int, ellipse[0])
    eyes_radius = int(max(ellipse[1]) / 4)
    cv.circle(circle, (x, y), eyes_radius, (255, 255, 255), cv.FILLED)
    cv.circle(circle, (x, y), eyes_radius // 2, (0, 0, 255), cv.FILLED)
    circle = cv.bitwise_and(circle, mask)
    cv.add(frame, circle, frame)


def draw_landmarks(frame, landmarks):
    if len(landmarks) == 68:
        draw_polyline(frame, landmarks, 36, 41, True, (255, 0, 0)) # left eye
        draw_polyline(frame, landmarks, 42, 47, True, (255, 0, 0)) # right eye


cap = cv.VideoCapture(0)
face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
facemark = cv.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')

assert cap.isOpened()


while True:
    ret, frame = cap.read()
    overlay = np.zeros_like(frame)

    if not ret:
        print('Err')
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)
    if len(faces) != 0:
        ret, landmarks = facemark.fit(frame, faces)
        for face_landmarks in landmarks:
            draw_landmarks(overlay, face_landmarks[0])
        faces[:,2:] += faces[:,:2]

    for face in faces:
        x1, y1, x2, y2 = face
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))

    # cv.addWeighted(overlay, 0.3, frame, 1, 0.7, frame)
    cv.blur(overlay, (3, 3), overlay)
    cv.add(frame, overlay, frame)
    cv.flip(frame, 1, frame)

    cv.imshow('frame', frame)

    if cv.waitKey(5) >= 0:
        break

cap.release()
cv.destroyAllWindows()
