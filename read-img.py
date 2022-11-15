import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0) # Clemens MacBook Pro 16
# cap = cv2.VideoCapture(?) # Beast Alpha


# Check if the webcam is opened correctly
while not cap.isOpened():
    print("Camera not found")
    raise IOError("Error: Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('OMD-D  E-M1', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()