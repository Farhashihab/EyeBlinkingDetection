import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def calmidpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def getBlinkingRation(points, facil_landmarks):
    left_point = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    right_point = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
    centerTop = calmidpoint(landmarks.part(points[1]), landmarks.part(points[2]))
    centerBottom = calmidpoint(landmarks.part(points[5]), landmarks.part(points[4]))

    h_line = cv2.line(frame, left_point, right_point, (0, 255, 00), 1)
    centerPoint = cv2.line(frame, centerTop, centerBottom, (0, 255, 0), 1)
    # cv2.circle(frame, (x, y), 3, (255, 0, 0,), -1)
    hor_line = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line = hypot((centerTop[0] - centerBottom[0]), (centerTop[1] - centerBottom[1]))
    # print(ver_line)
    ratio = hor_line / ver_line
    return ratio


count = 0
while True:

    __, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        leftEyeRation = getBlinkingRation([36, 37, 38, 39, 40, 41], landmarks)
        rightEyeRation = getBlinkingRation([42, 43, 44, 45, 46, 47], landmarks)
        blinkingration = (leftEyeRation + rightEyeRation) / 2
        # print(ratio)

        if blinkingration > 5.3:
            cv2.putText(frame, "BLINK", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 1)
            count += 1
        # print(face)
        # print(count)

        # Detect Gaze#

        leftEyeRegion = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                  (landmarks.part(37).x, landmarks.part(37).y),
                                  (landmarks.part(38).x, landmarks.part(38).y),
                                  (landmarks.part(39).x, landmarks.part(39).y),
                                  (landmarks.part(40).x, landmarks.part(40).y),
                                  (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        rightEyeRegion = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                   (landmarks.part(43).x, landmarks.part(43).y),
                                   (landmarks.part(44).x, landmarks.part(44).y),
                                   (landmarks.part(45).x, landmarks.part(45).y),
                                   (landmarks.part(46).x, landmarks.part(46).y),
                                   (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        cv2.polylines(frame, [leftEyeRegion], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEyeRegion], True, (0, 255, 0), 1)
        minX = np.min(leftEyeRegion[:, 0])
        maxX = np.max(leftEyeRegion[:, 0])
        minY = np.min(leftEyeRegion[:, 1])
        maxY = np.max(leftEyeRegion[:, 1])

        eye = frame[minY:maxY, minX:maxX]
        eye = cv2.resize(eye, None, fx=5, fy=5)

        eye= cv2.flip(eye,1)
        cv2.imshow("Eye", eye)
        frame= cv2.flip(frame,1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
print(count)
cap.release()
cv2.destroyAllWindows()
