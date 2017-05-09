import cv2
# Importing the opencv library
import imutils
# Importing the library that supports basic image processing functions
import numpy as np
# Importing the array operations library for python
import os
# Importing the library which supports standard systems commands
from scipy.cluster.vq import *
# Importing the library which classifies set of observations into clusters
from sklearn.preprocessing import StandardScaler
# Importing the library that supports centering and scaling vectors
from imutils import paths
import sys
sys.path.append('/home/pra/.virtualenvs/cv/lib/python2.7/site-packages/')
import matplotlib.pyplot as plt

SURF=cv2.xfeatures2d.SURF_create()
im=cv2.imread(sys.argv[1])
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(im,None)

# compute the descriptors with ORB
kp, des = orb.compute(im, kp)  # Computing the key points and the descriptors
print(kp,des)
plt.imshow(cv2.drawKeypoints(im, kp,im.copy()))
plt.show()
