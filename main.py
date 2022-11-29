import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from object_size import pixel_to_millimeters
from object_size import *

cap = cv2.VideoCapture(3)
time.sleep(0.5)

# Defining Color intervals to recognize
ORANGE_MIN = np.array([5, 50, 50],np.uint8)
ORANGE_MAX = np.array([30, 255, 255],np.uint8)
#define colors for size detection
BLACK_MIN = np.array([36, 50, 50],np.uint8)
BLACK_MAX = np.array([86, 255, 255],np.uint8)
#BLACK_MAX = np.array([110,50,50],np.uint8)
#BLACK_MIN = np.array([130,255,255],np.uint8)

GREEN_MIN = np.array([40, 40,40],np.uint8)
GREEN_MAX = np.array([70, 255,255],np.uint8)

# Defining the Coordinate Vectors for later analysis
X = []
Y = []
X_avg = []
Y_avg = []
T = []
time = 0

while True:
    _, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_MIN, GREEN_MAX)
    mask2 = cv2.inRange(hsv, BLACK_MIN, BLACK_MAX)
    
    #test zone
    #_, mask2 = cv2.threshold(hsv, 60, 255, cv2.THRESH_BINARY_INV)
    """
    masked = cv2.bitwise_and(frame.copy(), frame.copy(), mask2) 
    masked = frame.copy() - masked   
    cv2.imshow('mask2', masked)
    """


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fix the canny issue later:
    # _, frame_canny = cap.read() 
    # edges = cv2.Canny(frame, 100, 200)
    # print(type(edges))
    # cv2.imshow("CANNY", edges)
    # hsv_canny = cv2.cvtColor(edges, cv2.COLOR_BGR2HSV)
    # mask_canny = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
    # contours_canny, _ = cv2.findContours(mask_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    
    count = 0
    #print("-----------------------------------------")
    pixelsPerMetric = None

    for contour in contours:
        area = cv2.contourArea(contour)
        count += 1
        
        if area > 8000:
            X = []
            Y = []
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            dimension = contour.shape
            for i in range(0, dimension[0]):
                X.append(contour[i][0][0])
                Y.append(contour[i][0][1])
        #else:
            #print("Error: No contours or too many contours detected")

    for c in contours2:
        if cv2.contourArea(c) < 300:
            continue
        #print("test before function call")
        #frame, pixelsPerMetric = pixel_to_millimeters(frame,c, 10, pixelsPerMetric)
        #print(pixelsPerMetric)
    

    #for contour in contours_canny:
    #    area = cv2.contourArea(contour)
    #    if area > 8000:
    #        cv2.drawContours(frame_canny, contour, -1, (255, 0, 0), 3)
    #Vert = np.concatenate((frame, frame_canny), axis=0)
    #cv2.imshow('HASEL', Vert)

    # Capturing relevant data for plot
    if len(X) != 0 and len(Y) != 0:
        x_avg = sum(X)/len(X)
        y_avg = sum(Y)/len(Y)
        X_avg.append(x_avg)
        Y_avg.append(y_avg)
        T.append(time)
        time += 1


    cv2.imshow('HASEL', frame)

   
    
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows

plt.plot(T, Y_avg) 
plt.xlabel('Time') 
plt.ylabel('Strain (y-direction)') 
plt.title('HASEL Strain') 
plt.show() 
