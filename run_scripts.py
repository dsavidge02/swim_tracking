import os
import numpy as np

import filter_images
import kalman_tracking
import arm_tracking
import video_creation

if __name__ == '__main__':
	# Inputs to change
	person = "Mike"
	stroke = "Free"
	start_point = 103
	s1 = np.array([[584,1001]])
	s2 = np.array([[584,1002]])


	# Creating subfolders
	if not os.path.isdir("results"):
		os.makedirs("results")
	if not os.path.isdir("results"+"/"+person):
		os.makedirs("results"+"/"+person)
	if not os.path.isdir("results"+"/"+person+"/"+stroke):
		os.makedirs("results"+"/"+person+"/"+stroke)

	file_path = "images/"+person+"/"+stroke

	if not os.path.isdir(file_path+"/Filtered"):
		os.makedirs(file_path+"/Filtered")
	if not os.path.isdir(file_path+"/Filtered/Arms"):
		os.makedirs(file_path+"/Filtered/Arms")
	if not os.path.isdir(file_path+"/Filtered/Kalman"):
		os.makedirs(file_path+"/Filtered/Kalman")

	if not os.path.isdir(file_path+"/Tracked"):
		os.makedirs(file_path+"/Tracked")
	if not os.path.isdir(file_path+"/Tracked/Arms"):
		os.makedirs(file_path+"/Tracked/Arms")
	if not os.path.isdir(file_path+"/Tracked/Kalman"):
		os.makedirs(file_path+"/Tracked/Kalman")

	# Running each script
	# filter_images.run_script(person, stroke)
	# kalman_tracking.run_script(person, stroke, start_point, s1, s2)
	# arm_tracking.run_script(person, stroke, start_point)
	# video_creation.run_script(person, stroke)
