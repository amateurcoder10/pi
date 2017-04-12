import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
kernel = np.ones((3,3),np.uint8)
if len(sys.argv) != 2:
    print ("%s input_file output_file" % (sys.argv[0]))
    sys.exit()
else:
    img = cv2.imread(sys.argv[1]) #Gray scale
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.jpeg',img2)
    blur = cv2.GaussianBlur(img2,(5,5),0) #Guassian blurrring

    #Otsu thresholding
    ret3,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
    cv2.imwrite('Otsu.jpeg',th)

    #Dilation
    dilation = cv2.dilate(th,kernel,iterations = 5) 
    cv2.imwrite('Dil.jpeg',dilation)

    #finding contours
    cnt, contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    cv2.imwrite('Cnt.jpeg',cnt)

    '''
    plt.subplot(2,2,1),plt.imshow(img2,"gray")
    plt.title('Original'),plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow( blur,"gray")
    plt.title('After blurring'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow( th, "gray")
    plt.title('OTSU'),plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow( dilation, "gray")
    plt.title('Dilation'),plt.xticks([]), plt.yticks([])
    plt.show()
    '''
    k=cv2.waitKey(0)
    cv2.destroyAllWindows()

