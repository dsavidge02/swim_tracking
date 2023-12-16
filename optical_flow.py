from scipy.signal import convolve2d
import os
import matplotlib.pyplot as plt
import numpy as np

from helper import *


def lucasKanade(img1, img2, K, pnt1, pnt2):
	img1 = img1/255
	img2 = img2/255
	kern_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	kern_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	kern_t = np.array([[1, 1], [1, 1]])

	mode = 'same'
	boundary = 'symm'
	Ic = convolve2d(img1, kern_x, boundary=boundary, mode=mode)
	Ir = convolve2d(img1, kern_y, boundary=boundary, mode=mode)
	It = convolve2d(img2, kern_t, boundary=boundary, mode=mode)+convolve2d(img1, -kern_t, boundary=boundary,mode=mode)
	u = np.zeros(img1.shape)
	v = np.zeros(img1.shape)
	w = K//2

	upper_threshold = 100
	lower_threshold = 1e-10
	for i in range(w, img1.shape[0]-w):
		for j in range(w, img1.shape[1]-w):
			ic = Ic[i-w:i+w+1, j-w:j+w+1].flatten()
			ir = Ir[i-w:i+w+1, j-w:j+w+1].flatten()
			A = np.vstack((ic,ir)).T
			b = It[i-w:i+w+1, j-w:j+w+1].flatten()
			cond = np.linalg.cond(A)
			if cond > upper_threshold:
				u[i,j] = 0
				v[i,j] = 0
			else:
				cond = np.linalg.cond(A.T @ A)
				if cond > 50:
					u[i,j] = 0
					v[i,j] = 0
				else:
					ls = -(np.linalg.inv(A.T @ A) @ A.T @ b)
					if np.abs(ls[0]) < lower_threshold:
						u[i,j] = 0
					else:
						u[i,j] = ls[0]
					if np.abs(ls[1]) < lower_threshold:
						v[i,j] = 0
					else:
						v[i,j] = ls[1]

	u = [[pnt1[0]-pnt2[0]]]
	v = [[pnt1[1]-pnt2[1]]]
	return u, v


def find_opticalflow(images, points, K):
	num_images = len(images)
	if num_images != len(points[0]):
		print("Number of images does not match number of points")
		return
	optical_flows = []
	width = 1
	height = 1
	for i in range(num_images-1):
		# Calculate the bounding box around the point
		min_x = int(points[:,i][0])-width//2
		max_x = int(points[:,i][0])+width//2+1
		min_y = int(points[:,i][1])-height//2
		max_y = int(points[:,i][1])+height//2+1
		
		min_x2 = int(points[:,i+1][0])-width//2
		max_x2 = int(points[:,i+1][0])+width//2+1
		min_y2 = int(points[:,i+1][1])-height//2
		max_y2 = int(points[:,i+1][1])+height//2+1
		img1 = images[i][min_y:max_y,min_x:max_x]
		img2 = images[i+1][min_y2:max_y2,min_x2:max_x2]
		of = np.array(lucasKanade(img1, img2, K, points[:,i], points[:,i+1]))
		optical_flows.append(of)
	return optical_flows


def visualize_opticalflows(folder_to_read, image_folder_to_draw_to, ofs, points, K):
	images = read_images(folder_to_read, '.jpg', grayscale = False, unsorted = True, shift = 2)
	num_images = len(images)
	width = 1
	height = 1
	for i in range(1,num_images):
		of = ofs[i-1]
		img = images[i]

		min_x = int(points[:,i][0])-width//2
		max_x = int(points[:,i][0])+width//2+1
		min_y = int(points[:,i][1])-height//2
		max_y = int(points[:,i][1])+height//2+1

		u = of[0, ::K, ::K]
		v = of[1, ::K, ::K]
		ax = plt.gca()
		ax.imshow(images[i],cmap='gray')
		X,Y = np.mgrid[min_x:max_x:K, min_y:max_y:K]
		ax.quiver(X,Y,u,-v,color='r',scale=5)
		plt.axis('off')
		filename = f'{image_folder_to_draw_to}/frame{i-1}.png'
		plt.margins(0,0)
		plt.savefig(filename,bbox_inches='tight',dpi=300)
		plt.close()

#Reads images from a specified location and writes the optical flows between each frame into a file
#Returns the calculated optical flow and head point where the bounding box is created around
#-------------
#folder_to_read - location of the image files
#points_file_to_read - location of the head points file
#file_to_write - location of where you want to write the angles to
#start_point - frame to start from
def calc_opticalflows(folder_to_read, points_file_to_read, file_to_write = None, image_folder_to_draw_to = None,
	start_point=0, K=5):
	images = read_images(folder_to_read, '.jpg', grayscale = True, unsorted = True, shift = 2)
	images = images[start_point:]
	points = read_points_from_txt(points_file_to_read)
	all_ofs = find_opticalflow(images, points, K)
	# NEED TO IMPLEMENT IN HELPER FILE
	# if file_to_write is not None:
	# 	write_ofs(all_ofs, file_to_write)
	visualize_opticalflows(folder_to_read, image_folder_to_draw_to, all_ofs, points, K)



def run_script(person, stroke, start_point=None):
	folder_to_read = f'images/{person}/{stroke}'
	points_file_to_read = f'results/{person}/{stroke}/kalman_points.txt'
	file_to_write = f'results/{person}/{stroke}/optical_flows.txt'
	image_folder_to_draw_to = f'images/{person}/{stroke}/Tracked/OpticalFlows'

	if start_point is None:
		print("Please find a starting frame using Filtered Kalman images")
		sys.exit()

	K = 1
	calc_opticalflows(folder_to_read, points_file_to_read, file_to_write, image_folder_to_draw_to,
		start_point=start_point, K=K)

# For testing
run_script("Daytona", "Free", 101)