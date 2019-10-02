#########################################################################################

# Lip Detector Module
# Matt Rochford
 
# This file defines the function to be used for lip detection in videos.
# Input is a path to an mp4 file (or other video type supported by cv2.VideoCapture).
# Output is a 3D data packet of frames of lip data stored as a numpy array.

######################################################################################### 

# Import the necessary packages
from imutils import face_utils
import numpy as np
import dlib
import cv2

######################################################################################### 

# Define path to dlib predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat' 

######################################################################################### 

# Lip detection function
def lip_detector(video_path):

	# Initialize dlib's face detector (HOG-based) and create facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	# Read video in as mp4 file
	video = cv2.VideoCapture(video_path)

	# Check if video opened
	if (video.isOpened()==False):
		print('Error opening video file: ' + video_path)
		return # Return from function if video does not open

	lip_frames = [] # Initialize variable to store lip frames

	# Read video frame by frame
	while(video.isOpened()):

		ret, frame = video.read() # Capture frame
		if ret == True: # if frame exists

			# Convert frame to grayscale and resize to a standard size
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame = cv2.resize(frame,(224,224))

			# detect face in the image
			faces = detector(frame, 1)

			if len(faces) > 1: # If more than one face detected print error and return
				print('Error: Multiple faces detected in video')
				return
			elif len(faces) == 0: # If no face detected print error and return
				print('Error: No face detected in video')
				return

			else: # If one face detected perform lip cropping
				for face in faces:

					# Determine facial landmarks for the face region and convert to a NumPy array
					shape = predictor(frame, face)
					shape = face_utils.shape_to_np(shape)

					# Extract the lip region as a separate image
					(x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
					margin = 10 # Extra pixels to include around lips
					lips = frame[y-margin:y + h + margin, x-margin:x + w + margin]
					lips = cv2.resize(lips,(100,60))

					# Create a stack of extracted frames
					if len(lip_frames) == 0:
						lip_frames = lips
					else:
						lip_frames = np.dstack(((lip_frames),(lips)))


		else: # If no frame left break from loop
			break

	# Release video object
	video.release() 
	# Close any open windows
	cv2.destroyAllWindows()

	# Reshape array for CNN layer compatibility
	lip_frames = np.moveaxis(lip_frames,-1,0)

	return lip_frames

######################################################################################### 

# REFERENCES:

# Code for reading and editing video files is adapted from 'Learn OpenCV'
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

# Code for lip detection is based off Dlib library and an implementation from PyImageSearch
# https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/






 