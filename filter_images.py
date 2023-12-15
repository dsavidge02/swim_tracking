import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from helper import *

#Takes an image and filters out colors that aren't close to red
#Returns an image that is filtered for Red
#-------------
#image - physical image
def apply_color_filter_for_kalman(image):
	hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	lower_red = np.array([0,120,50])
	upper_red = np.array([10,255,255])
	mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
	lower_red2 = np.array([160,120,50])
	upper_red2 = np.array([180,255,255])
	mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
	combined_mask = mask1 | mask2
	result = cv2.bitwise_and(image, image, mask=combined_mask)
	result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
	return result

#Takes an image and filters out for a skin color range
#Returns an image that is filtered to the range of skin colors
#-----------
#image - physical image
def apply_arm_filter(image):
	hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	lower_yellow_green = np.array([0,50,80])
	upper_yellow_green = np.array([18,255,255])
	mask1 = cv2.inRange(hsv_image, lower_yellow_green, upper_yellow_green)
	combined_mask = mask1
	result = cv2.bitwise_and(image, image, mask=combined_mask)
	result  = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
	return result

#Takes and image and removes the lane lines from it
#Returns an image that is filtered to remove lane lines
#--------------------
#image - physical image
def remove_lane_lines(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(image, (x1, y1), (x2, y2), (0,0,0),10)
	return image

#Filters images for use in kalman filtering
#Returns none
#------------------
#images_to_filter - folder with the images to filter
#outer_path - outer folder to save to
#inner_folder - inner folder to save to
def filter_for_kalman(images_to_filter, outer_path, inner_folder):
	images = read_images(images_to_filter, '.jpg', grayscale = False, unsorted = True, shift = 2)
	filtered_images = []
	for image in images:
		filt_image = apply_color_filter_for_kalman(image)
		filtered_images.append(filt_image)
	save_images(outer_path, inner_folder, filtered_images)

#Filters images for use in arm tracking
#Returns none
#-------------------
#images_to_filter - folder with the images to filter
#outer_path - outer folder to save to
#inner_folder - inner folder to save to
def filter_for_arm_tracking(images_to_filter, outer_path, inner_folder):
	images = read_images(images_to_filter, '.jpg', grayscale = False, unsorted = True, shift = 2)
	filtered_images = []
	for image in images:
		filt_image = apply_arm_filter(image)
		filt_image = remove_lane_lines(filt_image)
		filtered_images.append(filt_image)
	save_images(outer_path, inner_folder, filtered_images)

#Filters for Kalman and Arm Tracking
def filter_all(images_to_filter, outer_path, inner_folder_kalman = 'Kalman', inner_folder_arm = 'Arms'):
	filter_for_kalman(images_to_filter, outer_path, inner_folder_kalman)
	filter_for_arm_tracking(images_to_filter, outer_path, inner_folder_arm)

#SAMPLE USAGE
person = 'Sierra'
stroke = 'Free'
images_to_filter = f'images/{person}/{stroke}'
outer_path = f'images/{person}/{stroke}/Filtered'
filter_all(images_to_filter, outer_path)