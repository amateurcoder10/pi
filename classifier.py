import argparse as ap
# Importing library that supports user friendly commandline interfaces
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
from imutils import paths
# Load the classifier, class names, scaler, number of clusters and vocabulary 
class patternlessException(Exception):
 
    # Constructor or Initializer
    def __init__(self):
     	pass

samples = np.loadtxt('samples.data',np.float32)
print(samples.shape)

responses = np.loadtxt('responses.data',np.float32)
print(responses.shape)
classes_names = np.load('training_names.data.npy')
print(classes_names)
voc = np.load('voc.data.npy')
#print(voc)
k = 500  # Loading the number of cluster

# Training the Knearest classifier with the test descriptors
clf = cv2.ml.KNearest_create()
clf.train(samples,cv2.ml.ROW_SAMPLE,responses)  # Train model using the training samples and corresponding responses

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
image_paths = []

if args["testingSet"]:
    test_path = args["testingSet"]
    print(test_path)
    try:
        testing_names = os.listdir(test_path)
	#print(testing_names)
    except OSError:
        print "No such directory {}\nCheck if the file exists".format(test_path)
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = list(paths.list_images(dir))
        #print(dir)
	#print(class_path)
        image_paths+=class_path
else:
    image_paths = [args["image"]]

#print(image_paths)
    
# Create feature extraction and keypoint detector objects
SURF=cv2.xfeatures2d.SURF_create()
l=0
# List where all the descriptors are stored
des_list = []
for image_path in image_paths:
    im = cv2.imread(image_path)
    if im == None:
        print "No such file {}\nCheck if the file exists".format(image_path)
        exit()
    #l=l+1
    #print(l)
    kpts, des =SURF.detectAndCompute(im, None)  # Computing the descriptors of the test image
    if(args["image"]):
	if(des==None):
		flag=1
		try:
			raise patternlessException
		except patternlessException as error:
			image = cv2.imread(image_path)
			cv2.namedWindow("Image",cv2.WINDOW_AUTOSIZE)  # Creating a named window
			pt = (1,image.shape[0]//6)  # Framing the size of font on the image
			cv2.putText(image, "patternless", pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1, [0, 0, 255],1)  # Placing text on the image
			cv2.imshow("Image",image)  # Displaying the image
			cv2.imwrite("res.jpg",image)
			cv2.waitKey()  # Wait for the keystroke of the user
			exit()    
    des_list.append((image_path, des))   # Appending the descriptors to a single list

print(len(des_list))


# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
   
    descriptors = np.vstack((descriptors, descriptor))   # Stacking the descriptors in to a numpy array

# Computing the histogram of features
test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1  # Calculating the histogram of features

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)  # Getting the number of occurrences of each word
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
# Assigning weight to one that is occurring more frequently

# Perform the predictions
retval, results, neigh_resp, dists = clf.findNearest(test_features,k=4)
# Finding the 17 nearest neighbours of the test image descriptor
#print(results,neigh_resp,dists)
#print(results)

l=0;
# Visualize the results, if "visualize" flag set to true by the user
if args["visualize"]:
	if  args["testingSet"]:
	    for image_path in image_paths:
		   

 		if results[l][0] == 0:  # results[0][0] will have the predicted class
	    		prediction = "plaid"
	    	elif results[l][0]==1:
			prediction = "striped"
	    	else:
			prediction="irregular"
		l=l+1
	    	image = cv2.imread(image_path)
		cv2.namedWindow("Image",cv2.WINDOW_AUTOSIZE)  # Creating a named window
		pt = (1,image.shape[0]//6)  # Framing the size of font on the image
		cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2, [0, 0, 255],2)  # Placing text on the image
		cv2.imshow("Image",image)  # Displaying the image
		cv2.waitKey()  # Wait for the keystroke of the user
	else:
            if results[0][0] == 0:  # results[0][0] will have the predicted class
	    	prediction = "plaid"
	    elif results[0][0]==1:
		prediction = "striped"
	    else:
		prediction="irregular"
		    
	    image = cv2.imread(image_path)
	    cv2.namedWindow("Image",cv2.WINDOW_AUTOSIZE)  # Creating a named window
	    pt = (1,image.shape[0]//4)  # Framing the size of font on the image
	    cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1, [0, 0, 255],2)  # Placing text on the image
            cv2.imshow("Image",image)  # Displaying the image
	    cv2.imwrite("res.jpg",image)
	    cv2.waitKey()  # Wait for the keystroke of the user

