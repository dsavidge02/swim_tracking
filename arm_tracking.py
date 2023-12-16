import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math

from helper import *

#Looks at a series of images and finds the angles between each point in points and the left and right arms
#Returns an array of all the angles from frame to frame as an array of [[left_angle, right_angle],...]
#------------------------------
#images - an array of all the images
#points - an array that is shape 2xN which contains the point of the head from frame to frame
def find_angles(images, points, debug = False):
	num_images = len(images)
	if num_images != len(points[0]):
		print("Number of images does not match number of points")
		return
	angles = []
	xscale = 0.1
	yscale = 0.05
	scale_increment = int(num_images/10)
	for i in range(num_images):
		if i > 0 and i % scale_increment == 0:
			xscale += .1
			yscale += .05
		img = images[i]
		c, r = int(points[0][i]),int(points[1][i])
		new_width = int(img.shape[1]*xscale)
		new_height = int(img.shape[0]*yscale)
		top_left_x = max(c-new_width//2, 0)
		top_left_y = max(r-new_height//2, 990)
		bottom_right_x = min(c+new_width//2,img.shape[1])
		bottom_right_y = min(r+new_height//2,img.shape[0])
		cropped_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
		c = c - top_left_x
		r = r - top_left_y
		binary_image = binarize_image_in_blocks(cropped_img)
		if debug:
			display_image(binary_image, True, c, r)
		labeled_image, num_labels = connected_components(binary_image)
		if debug:
			plt.imshow(labeled_image, cmap='jet')
			plt.colorbar()
			plt.show()
		labeled_image = filter_components_by_size(labeled_image, 25, 5000)
		left_label, right_label, left_centroid, right_centroid, d_l, d_r = find_closest_components(labeled_image, c, r)
		left_angle = 0.0
		right_angle = 0.0
		if left_label != None:
			left_angle = find_angle([c,r],left_centroid)
		if right_label != None:
			right_angle = find_angle([c,r], right_centroid)
		angles.append([left_angle, right_angle])
		if debug:
			mask = create_mask_from_components(labeled_image, [left_label, right_label])
			isolated_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)
			plt.imshow(labeled_image, cmap='jet')
			plt.colorbar()
			plt.show()
			print(f'left {d_l}')
			print(f'right {d_r}')
			display_image(isolated_image, True, c, r, left_centroid, right_centroid)
	return angles

#Takes an image and represents it as a series of black and white blocks
#Returns the new image in binary represetation - white 255, black 0
#------------------
#image - physical image
#block_size - represents an nxn block that will be converted to black if less than frac% of the block is black
def binarize_image_in_blocks(image, block_size=3, frac = .25):
    height, width = image.shape[:2]
    binary_image = np.zeros_like(image)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            non_zero = np.count_nonzero(block)
            ratio = non_zero / block_size**2
            if ratio > frac:
            	binary_image[y:y+block_size, x:x+block_size] = 255
            else:
            	binary_image[y:y+block_size, x:x+block_size] = 0
    return binary_image

#Implements the flood fill algorithm to label connected components in an image
#Modifies the label_image in place, marking connected regions with the same label
# ------------------
# label_image - a 2D array where the labels of connected components are stored
# image - the physical image
# x, y - starting x and y coordinates for the flood fill
# label - the label to assign to the connected component being filled
def flood_fill(label_image, image, x, y, label):
	stack = [(x,y)]
	while stack:
		x,y = stack.pop()
		if image[y,x] and label_image[y,x] == 0:
			label_image[y, x] = label
			if x > 0:
				stack.append((x-1,y))
			if x < image.shape[1]-1:
				stack.append((x+1,y))
			if y > 0:
				stack.append((x,y-1))
			if y < image.shape[0]-1:
				stack.append((x,y+1))

#Takes an image and labels the connected components
#Returns a 2d array of the same size where connected components are represented by a label number as well as the number of labels
#-----------------------------
#image - the physical image
def connected_components(image):
	labeled_image = np.zeros_like(image)
	label = 0
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			if image[y,x] and labeled_image[y,x] == 0:
				label += 1
				flood_fill(labeled_image, image, x, y, label)
	return labeled_image, label

#Takes a labeled image and filters out certain components based off of their size
#Returns a new labeled image that doesn't contain components that are too small or too big
#------------------------
#labeled_image - 2d array that represents the connected components by their label numbers
#min_size - the minimum number of pixels that can make up a component
#max_size - the maximum number of pixels that can make up a component
def filter_components_by_size(labeled_image, min_size, max_size):
	height, width = labeled_image.shape
	component_sizes = {}

	# Calculate the size of each component
	for y in range(height):
		for x in range(width):
			label = labeled_image[y, x]
			if label != 0:
				component_sizes[label] = component_sizes.get(label, 0) + 1

	# Filter out components that are too small or too large
	for label, size in component_sizes.items():
		# print(f'{label}: {size} {size < min_size} {size > max_size}')
		if size < min_size or size > max_size:
			# print('hi')
			labeled_image[labeled_image == label] = 0  # Set pixels of filtered components to 0

	return labeled_image

#Takes a labeled image and finds the two connected components that are closet to the c,r point
#Returns the labels, centroids, and distances of the closest components on the left and right of the point
#---------------------------------
#labeled_image - 2d array that represents the connected components by their label numbers
#c - c coordinate of point
#r - r coordinate of point
def find_closest_components(labeled_image, c, r):
    height, width = labeled_image.shape
    component_centroids = {}
    component_pixel_counts = {}

    # Calculate centroids for each component
    for y in range(height):
        for x in range(width):
            label = labeled_image[y, x]
            if label != 0:
                if label not in component_centroids:
                    component_centroids[label] = [0, 0]
                    component_pixel_counts[label] = 0
                component_centroids[label][0] += x
                component_centroids[label][1] += y
                component_pixel_counts[label] += 1

    for label in component_centroids.keys():
        component_centroids[label][0] /= component_pixel_counts[label]
        component_centroids[label][1] /= component_pixel_counts[label]

    # Find closest components to the left and right
    closest_left_label, closest_right_label = None, None
    closest_left_distance, closest_right_distance = float('inf'), float('inf')
    left_centroid = [-1,-1]
    right_centroid = [-1,-1]
    for label, centroid in component_centroids.items():
        distance = math.sqrt((centroid[0] - c) ** 2 + (centroid[1] - r) ** 2)
        distance_to_top = centroid[1]
        # distance_to_bottom = height-centroid[1]
        # distance_to_side = min(centroid[0], width-centroid[0])
        if distance_to_top < 20: #or distance > distance_to_bottom or distance > distance_to_side:
        	continue
        if distance > 50:
        	continue
        if centroid[0] < c and distance < closest_left_distance:
            closest_left_label, closest_left_distance = label, distance
            left_centroid = [centroid[0],centroid[1]]
        elif centroid[0] > c and distance < closest_right_distance:
            closest_right_label, closest_right_distance = label, distance
            right_centroid = [centroid[0],centroid[1]]

    return closest_left_label, closest_right_label, left_centroid, right_centroid, closest_left_distance, closest_right_distance

#Takes a labeled image and the the desired labels of the components to keep and creates a mask of that image
#Returns a mask of the image
#-------------------------
#labeled_image - 2d array that represents the connected components by their label numbers
#components - an array of the labels of the components that are to be kept after applying the mask
def create_mask_from_components(labeled_image, components):
	mask = np.zeros_like(labeled_image)
	for label in components:
		if label is not None:
			mask[labeled_image == label] = 255
	return mask

#Takes the point of the head and the point of an arm and calculates the angle between the two
#Returns the angle in radians between the head and the arm
#-----------
#head - point of the head
#arm - point of the arm
def find_angle(head, arm):
	x1, y1 = head
	x2, y2 = arm
	if arm[0] == -1 or arm[1] == -1:
		return 0.0
	adjacent = x2-x1
	hypotenuse = math.sqrt((x2-x1)**2+(y2-y1)**2)
	angle_rad = math.acos(adjacent/hypotenuse)
	return angle_rad

#Takes the list of angles and filters it for outliers
#Returns an updated array of angles
#-------------
#angles - [left_angle, right_angle] angle measures in radians
def update_angles(angles):
	num_angles = len(angles)
	for i in range(1, num_angles-1):
		left_angle_prev,right_angle_prev = angles[i-1]
		left_angle,right_angle = angles[i]
		left_angle_post,right_angle_post = angles[i+1]
		if not(1.5078 <= left_angle <=4.71239):
			left_angle = 0.0
		if not(right_angle <= 1.5078 or right_angle > 4.71239):
			right_angle = 0.0
		if left_angle_prev == 0.0 and left_angle_post == 0.0:
			left_angle = 0.0
		if right_angle_prev == 0.0 and right_angle_post == 0.0:
			right_angle = 0.0
		angles[i] = [left_angle,right_angle]
	return angles

#Takes an image and draws on it the lines between a point on the head and a point on the arms
#Returns none
#-------------
#image - physical image
#start_point - [x,y] coordinates of the head
#angles - [left_angle, right_angle] angle measures in radians
#length - length of the line
#thickness - thickness of the line
#filename - the name of the file that is to be saved or false if you want to display the image
def draw_angles(image, start_point, angles, length=100, thickness=10, filename = False):
    left_angle, right_angle = angles
    left_end_x = start_point[0] + length * math.cos(left_angle)
    left_end_y = start_point[1] - length * math.sin(left_angle)  # Subtract for y as y increases downwards
    right_end_x = start_point[0] + length * math.cos(right_angle)
    right_end_y = start_point[1] - length * math.sin(right_angle)

    image_with_lines = image.copy()
    if left_angle != 0:
    	image_with_lines = cv2.line(image, start_point, (int(left_end_x), int(left_end_y)), (255, 0, 0), thickness)  # Blue line for left
    if right_angle != 0:
    	image_with_lines = cv2.line(image_with_lines, start_point, (int(right_end_x), int(right_end_y)), (0, 255, 0), thickness)  # Green line for right    

    if filename:
    	image_cvt = cv2.cvtColor(image_with_lines, cv2.COLOR_RGB2BGR)
    	cv2.imwrite(filename, image_cvt)
    else:
    	plt.imshow(image_with_lines)
    	plt.show()

#Writes the angles down into a file 
#Returns none
#--------------
#angles - array of angles [[left_angle, right_angle],...]
#file_path - folder/file.txt to write to
def write_angles(angles, file_path):
	with open(file_path, 'w') as file:
		for left_angle, right_angle in angles:
			file.write(f'{left_angle}, {right_angle}\n')

#Reads the angles from a file into an array
#Returns an array of angles [[left_angle, right_angle],...]
#------------
#file_path - folder/file.txt to write to
def read_angles_from_file(file_path):
	angles = []
	with open(file_path, 'r') as file:
		for line in file:
			left_angle, right_angle = map(float, line.strip().split(','))
			angles.append([left_angle, right_angle])
	return angles

#Reads filtered images from a specified location and writes the angles between the left and right arm into a file
#Returns the measured angles and the head points used to calculate them
#-------------
#filtered_folder_to_read - location of the filtered files
#points_file_to_read - location of the head points file
#file_to_write - location of where you want to write the angles to
#start_point - frame to start from
def determine_angles(filtered_folder_to_read, points_file_to_read, file_to_write = None, start_point = 0):
	images = read_images(filtered_folder_to_read, '.jpg', grayscale = True, unsorted = True)
	images = images[start_point:]
	points = read_points_from_txt(points_file_to_read)
	all_angles = find_angles(images, points)
	all_angles = update_angles(all_angles)
	if file_to_write:
		write_angles(all_angles, file_to_write)
	return all_angles, points

#Draws the angles on the images for visual analysis
#Returns none
#------------------
#image_folder_to_draw_on - location of the images to draw on
#image_folder_to_draw_to - location of where to put the drawn images
#points_file_to_read - location of the head points file
#points - location of the head points
#angles_file_to_read - location of the angles file
#all_angles - location of the head points
#start_point - frame to start from
def visualize_all_angles(image_folder_to_draw_on, image_folder_to_draw_to = False, points_file_to_read = '', points = [],  angles_file_to_read = '', all_angles = [], start_point = 0):
	images = read_images(image_folder_to_draw_on, '.jpg', grayscale = False, unsorted = True, shift = 2)
	images = images[start_point:]
	if points_file_to_read != '':
		points = read_points_from_txt(points_file_to_read)
	elif len(points) == 0:
		print('No head points given')
		return
	if angles_file_to_read != '':
		all_angles = read_angles_from_file(angles_file_to_read)
	elif len(all_angles) == 0:
		print('No angles given')
		return
	for i in range(len(images)):
		img = images[i]
		point = int(points[0][i]), int(points[1][i])
		angles = all_angles[i]
		if image_folder_to_draw_to:
			filename = f'{image_folder_to_draw_to}/frame{i}.png'
			draw_angles(img, point, angles, filename = filename)
		else:
			draw_angles(img, point, angles, filename = image_folder_to_draw_to)


#Runs determine_angles and visualize_all_angles
def track_arms(filtered_folder_to_read, points_file_to_read, file_to_write, image_folder_to_draw_on, image_folder_to_draw_to, start_point = 0):
	all_angles, points = determine_angles(filtered_folder_to_read, points_file_to_read, file_to_write = file_to_write, start_point = start_point)
	visualize_all_angles(image_folder_to_draw_on, image_folder_to_draw_to  = image_folder_to_draw_to, points = points, all_angles = all_angles, start_point = start_point)

def run_script(person, stroke, start_point=None):
	filtered_folder_to_read = f'images/{person}/{stroke}/Filtered/Arms'
	points_file_to_read = f'results/{person}/{stroke}/kalman_points.txt'
	file_to_write = f'results/{person}/{stroke}/arm_angles.txt'
	image_folder_to_draw_on = f'images/{person}/{stroke}'
	image_folder_to_draw_to = f'images/{person}/{stroke}/Tracked/Arms'

	if start_point is None:
		print("Please find a starting frame using Filtered Kalman images")
		sys.exit()
	track_arms(filtered_folder_to_read, points_file_to_read, file_to_write, image_folder_to_draw_on, image_folder_to_draw_to, start_point = start_point)

# #SAMPLE USAGE
# person = 'Sierra'
# stroke = 'Free'
# filtered_folder_to_read = f'images/{person}/{stroke}/Filtered/Arms'
# points_file_to_read = f'results/{person}/{stroke}/kalman_points.txt'
# file_to_write = f'results/{person}/{stroke}/arm_angles.txt'
# image_folder_to_draw_on = f'images/{person}/{stroke}'
# image_folder_to_draw_to = f'images/{person}/{stroke}/Tracked/Arms'
# start_point = 107
# track_arms(filtered_folder_to_read, points_file_to_read, file_to_write, image_folder_to_draw_on, image_folder_to_draw_to, start_point = start_point)
