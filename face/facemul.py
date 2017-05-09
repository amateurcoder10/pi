import numpy as np
import cv2
import math
from PIL import Image
import os, os.path


face_cascade = cv2.CascadeClassifier('/home/pra/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('/home/pra/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml')

imgs = []
path = "/home/pra/2sem/mp/project/face/faces"
valid_images = [".jpg",".jpeg",".png"]

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(cv2.imread(os.path.join(path,f)))

print(len(imgs))

for img in imgs:
	#img = cv2.imread('img.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print faces
    print("the no of face is ",len(faces))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	   
    cv2.imshow('img',img)
    cv2.waitKey(0)

