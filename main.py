import cv2
import matplotlib.pyplot as plt
import numpy as np

fileName = "img/img-1.png"
im1 = cv2.imread(fileName, cv2.IMREAD_COLOR)
im2 = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)


plt.figure(figsize=[20,10])
plt.subplot (121); plt.axis('off'); plt.imshow(im1); plt.title("COLOR")
plt.subplot (122); plt.axis('off'); plt.imshow(im2); plt.title( "BLACKWHITE")
plt.show