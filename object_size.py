#reference https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
#from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def pixel_to_millimeters(frame, c, ref_height, pixelsPerMetric): #reference height in millimeters
    #create box
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
    # loop over the original points and draw them (points as small circles)
    for (x, y) in box:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr) #top midpoint
    (blbrX, blbrY) = midpoint(bl, br) #bottom midpoint
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl) #right midpoint
    (trbrX, trbrY) = midpoint(tr, br) #left midpoint
    # draw the midpoints on the image
    cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # draw lines between the midpoints
    cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), #top to bottom
        (255, 0, 255), 2)
    cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), #left to right
        (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric == None:
        pixelsPerMetric = dA / ref_height
        #print(pixelsPerMetric)

    return frame, pixelsPerMetric
