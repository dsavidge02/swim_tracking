import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys

from helper import *

#Generate all of the initial parameters needed for Kalman Filtering
def initial_state_vector(points_0, points_1):
    s_0s = []
    for i in range(len(points_0)):
        c_0 = points_1[i][0]
        r_0 = points_1[i][1]
        v_c0 = c_0 - points_0[i][0]
        v_r0 = r_0 - points_0[i][1]
        s_0 = np.array([c_0,r_0,v_c0, v_r0])
        s_0s.append(s_0)
    return np.array(s_0s)

def deconstruct_state_vector(sv):
    points = []
    for v in sv:
        point = [v[0],v[1]]
        points.append(point)
    return np.array(points)

def initial_state_covariance():
    sigma_0 = np.array([[100,0,0,0],[0,100,0,0],[0,0,25,0],[0,0,0,25]])
    return sigma_0

def state_covariance():
    Q = np.array([[16,0,0,0],[0,16,0,0],[0,0,4,0],[0,0,0,4]])
    return Q

def measurement_covariance():
    R = np.array([[4,0],[0,4]])
    return R

def state_transition_matrix():
    Phi = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    return Phi

def measurement_matrix():
    H = np.array([[1,0,0,0],[0,1,0,0]])
    return H

#s1 and s2 are the initial state vectorss
def initialize_Kalman_parameters(s1, s2):
	sv_0 = initial_state_vector(s1, s2)
	sigma_0 = initial_state_covariance()
	Q = state_covariance()
	R = measurement_covariance()
	Phi  = state_transition_matrix()
	H = measurement_matrix()
	return sv_0, sigma_0, Q, R, Phi, H

#Start of the Kalman Filter Class
class KalmanFilter:
	#initialize the Kalman Filter
	def __init__(self, F, H, Q, R, P, x, init_patch):
		self.F = F #Phi
		self.H = H
		self.Q = Q
		self.R = R
		self.P = P #Covariance Matrix
		self.x = x #State vector
		self.P_prev = P
		self.x_prev = x
		self.curr_patch = init_patch
		self.original_patch = init_patch
		self.P_s = [P]
		self.extend = 0

	#Predict the next measurement
	def predict(self):
		self.x_prev = self.x
		self.P_prev = self.P
		self.x = np.dot(self.F, self.x)
		self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
		return self.x

	#Unpredict the last measurement and extend the search range if the search was unsuccessful
	def unpredict(self):
		self.x = self.x_prev
		self.P = self.P_prev
		self.extend += 20

	#Update based off of the true measurement
	def update(self, z):
		y = z-np.dot(self.H, self.x)
		S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
		K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
		self.x = self.x + np.dot(K, y)
		self.P = self.P - np.dot(np.dot(K, S), K.T)
		self.P_s.append(self.P)

    #Used to find the true measurement location of the cap
	def find_cap(self, image, initial_patch, patch_size):
		predicted_position = (int(self.x[0]), int(self.x[1]))
		diag_c = self.P[0][0]
		diag_r = self.P[1][1]
		search_radius_row = int(2 * np.sqrt(diag_c))
		search_radius_col = int(2 * np.sqrt(diag_r))
		sum_x = 0
		sum_y = 0
		count = 0
		for c in range(predicted_position[0]-search_radius_col, predicted_position[0] + search_radius_col + 1):
			for r in range(predicted_position[1] - search_radius_row, predicted_position[1] + search_radius_row + 1 + self.extend):
				if image[r,c] > 5:
					sum_x += c
					sum_y += r
					count += 1
		if count > 2:
			center_x = sum_x // count
			center_y = sum_y // count
			self.extend = 0
			return [center_x, center_y], True
		else:
			return None, False

    #Track the point in the next frame
	def track_next_point(self, image, patch_size, frame_num, update_patch = False):
		predicted_position = self.predict()
		predicted_coords = np.array([predicted_position[0],predicted_position[1]])
		actual_position, found = self.find_cap(image, self.curr_patch, patch_size)
		actual_position = np.array(actual_position)
		if found:
			self.update(actual_position)
			if update_patch:
			    updated_patch = get_patch(image, actual_position, patch_size)
			    self.curr_patch = updated_patch
			return actual_position
		else:
			self.unpredict()
			return predicted_coords

#Get the patch covered by the size of the patch centered at the center
def get_patch(image, center, size):
        c, r = center
        half_size = size//2
        return image[r - half_size:r + half_size + 1, c - half_size:c + half_size + 1]

#Get the patches from the initial frames
def get_initial_patches(image, points, size):
    patches = []
    for point in points:
        patch = get_patch(image, point, size)
        patches.append(patch)
    return patches

#Track the points across all frames
def track_all_points(kfs, images, patch_size, update_patch = False, dynamic_patch_size = True):
    all_points = []
    i = 2
    for image in images[2:]:
        frame_points = []
        for kf in kfs:
            pos = kf.track_next_point(image, patch_size, i,update_patch)
            frame_points.append(pos)
        frame_points = np.array(frame_points)
        all_points.append(frame_points)
        i+=1
    all_points = np.array(all_points)
    return all_points


#Write the tracked points across the frames
def write_tracked_points(s1, s2, all_frames, file_path):
    output_string = ''
    for point in s1:
        c,r = point
        output_string = f'{output_string}{c},{r}\n'
    for point in s2:
        c,r = point
        output_string = f'{output_string}{c},{r}\n'
    for frame in all_frames:
        for point in frame:
            c,r = point
            output_string = f'{output_string}{c},{r}\n'
    file_loc = f'{file_path}/kalman_points.txt'
    write_file(file_loc, output_string)

#Used to locate pixels in first two frames
def choose_starting_pixels(images):
	display_image(images[0])
	display_image(images[1])
	sys.exit()

def run_script(person, stroke, start_point=None, s1=None, s2=None):
	images_to_track = f'images/{person}/{stroke}/Filtered/Kalman'
	images_to_overlay = f'images/{person}/{stroke}'
	image_save_loc = f'images/{person}/{stroke}/Tracked/Kalman/'
	write_points_loc = f'results/{person}/{stroke}'
	images = read_images(images_to_track, extension = '.jpg', grayscale = True, unsorted = True)
	if start_point == None:
		print("Please find a starting frame using Filtered Kalman images")
		sys.exit()
	images = images[start_point:]
	if s1 is None or s2 is None:
		print("Choosing a starting pixel from the first 2 frames")
		choose_starting_pixels(images)

	sv_0, sigma_0, Q, R, Phi, H = initialize_Kalman_parameters(s1, s2)
	patch_size = 5
	kalman_filters = []
	init_patches = get_initial_patches(images[0], s1, patch_size)
	for i, s_0 in enumerate(sv_0):
		kf = KalmanFilter(Phi,H,Q,R,sigma_0, s_0, init_patches[i])
		kalman_filters.append(kf)
	all_tracked_points = track_all_points(kalman_filters, images, patch_size, update_patch = True, dynamic_patch_size = False)
	write_tracked_points(s1, s2, all_tracked_points, write_points_loc)

	i = 2
	images = read_images(images_to_overlay, extension = '.jpg', grayscale = False, unsorted = True, shift = 2)
	images = images[start_point:]
	overlay_points(images[0], s1, grayscale = False, file_loc = image_save_loc, filename = 'frame0')
	overlay_points(images[1], s2, grayscale = False, file_loc = image_save_loc, filename = 'frame1')
	for image, points in zip(images[2:], all_tracked_points):
		img_str = f'frame{i}'
		overlay_points(image, points, grayscale = False, file_loc = image_save_loc, filename = img_str)
		i+=1



# #SAMPLE USAGE
# person = 'Sierra'
# stroke = 'Free'
# images_to_track = f'images/{person}/{stroke}/Filtered/Kalman'
# start_point = 107
# images_to_overlay = f'images/{person}/{stroke}'
# image_save_loc = f'images/{person}/{stroke}/Tracked/Kalman/'
# write_points_loc = f'results/{person}/{stroke}'

# #FIRST NEED TO LOCATE THE PIXELS IN THE FIRST TWO DESIRED FRAMES
# images = read_images(images_to_track, extension = '.jpg', grayscale = True, unsorted = True)
# images = images[start_point:]

# #COMMENT OUT ONCE POINTS ARE PICKED
# #choose_starting_pixels(images)

# #ONCE POINTS ARE PICKED REPLACE
# s1 = np.array([[547,1022]])
# s2 = np.array([[547,1023]])

# sv_0, sigma_0, Q, R, Phi, H = initialize_Kalman_parameters(s1, s2)
# patch_size = 5
# kalman_filters = []
# init_patches = get_initial_patches(images[0], s1, patch_size)
# for i, s_0 in enumerate(sv_0):
# 	kf = KalmanFilter(Phi,H,Q,R,sigma_0, s_0, init_patches[i])
# 	kalman_filters.append(kf)
# all_tracked_points = track_all_points(kalman_filters, images, patch_size, update_patch = True, dynamic_patch_size = False)
# write_tracked_points(s1, s2, all_tracked_points, write_points_loc)

# i = 2
# images = read_images(images_to_overlay, extension = '.jpg', grayscale = False, unsorted = True, shift = 2)
# images = images[start_point:]
# overlay_points(images[0], s1, grayscale = False, file_loc = image_save_loc, filename = 'frame0')
# overlay_points(images[1], s2, grayscale = False, file_loc = image_save_loc, filename = 'frame1')
# for image, points in zip(images[2:], all_tracked_points):
# 	img_str = f'frame{i}'
# 	overlay_points(image, points, grayscale = False, file_loc = image_save_loc, filename = img_str)
# 	i+=1