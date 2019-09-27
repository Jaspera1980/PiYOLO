#libraries
import cv2
import json
import datetime



#import variables
conf = json.load(open("conf.json"))

#Camera object and raw camera capture
camera = cv2.VideoCapture(0)
#alternative ipcamera
#camera = cv2.VideoCapture(conf["ipcamera"])
#alternative Raspberry Pi camera
#camera = PiCamera()

#Camera settings (for Raspberry Pi Camera)
# camera.resolution = tuple(conf["resolution"])
# camera.framerate = conf["fps"]
# rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

#Camera settings (for ipcamera)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, tuple(conf["resolution"])[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, tuple(conf["resolution"])[1])

#initialize the average frame
avg = None

#open camera
#loop
while True:
	ret, frame = camera.read()
	timestamp = datetime.datetime.now()
	text = "No detection"

	#resize the frame to greyscale and blur it
	# frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# show images
	cv2.imshow('frame', gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


camera.release()

cv2.destroyAllWindows()
