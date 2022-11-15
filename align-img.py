import cv2
import matplotlib.pyplot as plt
import numpy as np

fileName = "img/calib-1.png"
im1 = cv2.imread(fileName, cv2.IMREAD_COLOR)
im2 = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

#cv2.imshow('image',im1_gray)
#cv2.waitKey()
#cv2.imshow('image',im2)
#cv2.waitKey()

# Detect ORB features and compute descriptors.
MAX_NUM_FEATURES = 500
orb = cv2.ORB_create(MAX_NUM_FEATURES)
keypoints1, descriptors = orb.detectAndCompute(im1_gray, None)

# Display

X = []
Y = []
for i in range(0, len(keypoints1)):
    
    X.append(keypoints1[i].pt[0])
    Y.append(keypoints1[i].pt[1])
    #print(keypoints1[i].pt)
    #print(X[0][0])


plt.scatter(X, Y, s=30,marker='x',color='r') 
plt.show()


im1_display = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(139,0,0))


cv2.imshow('image',im1_display)
cv2.waitKey()


