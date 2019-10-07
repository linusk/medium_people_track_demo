# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import xapi_spi
from gpiozero import LED

def image_resize(image, width = None, height = None, inter = cv2.INTER_NEAREST):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    new_ratio = float(width / height)
    old_ratio = float(w / h)

    if old_ratio > new_ratio:
        r = width / float(w)
        dim = (width, int(h * r))
    elif old_ratio < new_ratio:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        dim = (width, height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    old_size = resized.shape[:2] # old_size is in (height, width) format

    delta_w = width - old_size[1]
    delta_h = height - old_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    #color = [0, 0, 0]
    color = [255, 255, 255]
    resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # return the resized image
    return resized

classname = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

font                   = cv2.FONT_HERSHEY_PLAIN
TopLeftCornerOfText = (10,11)
fontScale              = 1
fontColor              = (165,26,26)
lineType               = 1




# Reset K210
print("Reset K210")
k210_reset = LED(27)
k210_reset.off()
time.sleep(0.5)
k210_reset.on()
time.sleep(0.5)
print("Reset K210 .... Done")

# Initialize SPI
xapispi = xapi_spi.Xapi_spi(0,0,60000000)
xapispi.init()



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(10)
#(H, W) = (None, None)
(H, W) = (224, 320)

# load our serialized model from disk
#print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0, resolution=(640, 448)).start()
#vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	origframe = vs.read()
	#frame = imutils.resize(frame, width=400)
	frame = image_resize(origframe, width=W, height=H)


	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

 
	rects = []

	xapispi.spi_send_img(frame)
	boxes = xapispi.spi_getbox()

	if len(boxes) > 0:
          for box in boxes:
            x1 = box.x1*2
            x2 = box.x2*2
            y1 = box.y1*2
            y2 = box.y2*2
            boxclass = box.boxclass
            prob = box.prob
            #if model_def == 'voc':
            text = "{} : {:.2f}".format(classname[boxclass[0]],prob[0])
            #else:
            #text = "{:.2f}".format(prob[0])
            if prob[0] > 0.7 and boxclass[0] == 14 and prob[0] > 0.8 and ((x2-x1)<640/2):
              cv2.putText(origframe, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,fontColor, 1)
              cv2.rectangle(origframe, (x1, y1), (x2, y2), fontColor, 1)
              rects.append([int(x1),int(y1),int(x2),int(y2)])



	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	objcnt = 0
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		objcnt += 1
		text = "ID {}".format(objectID)
		cv2.putText(origframe, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(origframe, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	text = "Count {}".format(objcnt)
	cv2.putText(origframe, text, (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# show the output frame
	cv2.imshow("Frame", origframe)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
