# import the necessary packages
import numpy as np
import time
import cv2
import os
import credentials



dir = 'Images/'
#Video capture object
cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture(credentials.ipcamera)

#manual args
YOLO_path = 'YOLO_model'
#YOLO_path = 'YOLO_tiny_model'
confidence = 0.5
threshold =0.3
# The duration in seconds of the video captured
capture_duration = 30
person = False

args = {'yolo':YOLO_path, 'confidence':confidence, 'threshold':threshold}

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Recording object
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))


#loop
while True:
	ret, frame = cam.read()

	# if person == True:
	# 	person = False
	# 	#if pi record video for x secs
	# 	start_time = time.time()
	# 	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	# 	out = cv2.VideoWriter(dir + 'video_' + str(time.ctime()) + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (frame_width, frame_height))
	# 	print('recording video')
	# 	while( int(time.time() - start_time) < capture_duration ):
	# 		ret, frame = cam.read()
	# 		if ret == True:
	# 			out.write(frame)
	# 		else:
	# 			print('recording finished')
	# 			break



	cv2.imwrite(dir+ 'img.png', frame)
	# load our input image and grab its spatial dimensions
	image = cv2.imread(dir + 'img.png')
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			# #if person is detected
			if LABELS[classIDs[i]] == 'person':
				person = True
				#save image if person is detected
				cv2.imwrite(dir + 'img_' + str(time.ctime()) + '.png', image)
				#if pi record video for x secs
				start_time = time.time()
				# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
				out = cv2.VideoWriter(dir + 'video_' + str(time.ctime()) + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (frame_width, frame_height))
				print('recording video')
				while( int(time.time() - start_time) < capture_duration ):
					ret, frame = cam.read()
					if ret == True:
						out.write(frame)
					else:
						print('recording finished')
						break


	# show images
	cv2.imshow('image', image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()

cv2.destroyAllWindows()
