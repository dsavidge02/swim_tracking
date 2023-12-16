import os
import numpy as np

import filter_images
import kalman_tracking
import arm_tracking
import video_creation

if __name__ == '__main__':
	# Inputs to change
	person = "Hayden"
	stroke = "Free"
	start_point = 112
	# s1 = None
	# s2 = None
	s1 = np.array([[542,1035]])
	s2 = np.array([[542,1036]])


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
	print("Running Kalman Filtering")
	kalman_tracking.run_script(person, stroke, start_point, s1, s2)
	print("Running Arm Tracking")
	arm_tracking.run_script(person, stroke, start_point)
	print("Creating Videos")
	video_creation.run_script(person, stroke)
