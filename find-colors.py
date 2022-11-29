import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(3)
time.sleep(0.5)

# Defining Color intervals to recognize
ORANGE_MIN = np.array([5, 50, 50],np.uint8)
ORANGE_MAX = np.array([15, 255, 255],np.uint8)

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
    mask = cv2.inRange(hsv, (0, 40,40), (20,255,255))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 
    count = 0
    print("-----------------------------------------")
    
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
        else:
            print("Error: No contours or too many contours detected")

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

