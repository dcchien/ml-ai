##
from skimage.color import rgb2gray		#pip install scikit-image
import numpy as np
import cv2								#pip install opencv-python
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import ndimage

image = plt.imread('images/me.jpg')
print("Coloar image, shape height/width/rbg(3)", image.shape)

plt.imshow(image)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

gray_image = rgb2gray(image)
print("Coloar image, shape height/width", gray_image.shape)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.imshow(gray_image, cmap='gray')
plt.show()

###
"""
The height and width of the image is 1632 and 2070 respectively. 
We will take the mean of the pixel values and use that as a threshold. 
If the pixel value is more than our threshold, we can say that it belongs to an object. 
If the pixel value is less than the threshold, it will be treated as the background.

The darker region (black) represents the background and the brighter (white) region is the foreground.
"""
gray_r = gray_image.reshape(gray_image.shape[0]*gray_image.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray_image = gray_r.reshape(gray_image.shape[0],gray_image.shape[1])
print("Gray image matrix shape height/width", gray_image.shape)

plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
