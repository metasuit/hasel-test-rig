#Reference: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

#Import libraries
from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

#Define filenames
imgFileName = 'img/align-test-1.png'
refFileName = 'img/reference.png'

def alignImages(im1, im2):
    
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    list(matches).sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imshow('matches',imMatches)
    cv2.waitKey()

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h
 
if __name__ == '__main__':
    #Assign images
    ref = cv2.imread(refFileName, cv2.IMREAD_COLOR)
    img = cv2.imread(imgFileName, cv2.IMREAD_COLOR)
    
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(img, ref)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    cv2.imshow('output image',imReg)
    cv2.waitKey()

    # Print estimated homography
    print("Estimated homography : \n",  h)






"""
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
"""






"""
#USELESSSSSSSS

import matplotlib.pyplot as plt



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


"""