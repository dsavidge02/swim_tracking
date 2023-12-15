import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

#Reads in all images from a folder
#Returns an array of images
#--------------------------------
#file_path - folder location
#extension - image type
#grayscale - image is gray
#unsorted - folder is not mathematically sorted
#prefix - add on for the filename that preceeds image
#suffix - add on for the filename that follows the image number
#shift - used to designate the number of files in a folder that aren't images
#convert - does the file need to be converted to RGB or not
def read_images(file_path = 'seq', extension = '.pgm', grayscale = True, unsorted = False, prefix = '', suffix='', shift = 0, convert = True):
    image_directory = file_path
    images = []
    if unsorted:
        img_num = len(os.listdir(image_directory))-shift
        for i in range(0, img_num):
            filename = f'{prefix}frame{i}{suffix}{extension}'
            if grayscale:
                img = cv2.imread(os.path.join(image_directory,filename), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(os.path.join(image_directory,filename))
                if convert:
                	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)
    else:
        for filename in os.listdir(image_directory):
        #print(filename)
            if filename.endswith(extension):
                if grayscale:
                    img = cv2.imread(os.path.join(image_directory,filename), cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(os.path.join(image_directory,filename))
                    if convert:
                    	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is not None:
                    images.append(img)
    return images

#Displays a single image in a matplotlib plot
#Returns nothing
#--------------------------------------------
#image - physical image
#grayscale - image is gray
#c - c coordinate of pixel that is to be displayed
#r - r coordinate of pixel that is to be displayed
#left_centroid - coordinates of pixel to the left of the c,r pixel
#right_centroid - coordinates of pixel to the right of the c,r pixel
def display_image(image, grayscale = True, c = 0, r = 0, left_centroid = [-1,-1], right_centroid = [-1,-1]):
	if grayscale:
		plt.imshow(image, cmap = 'gray')
	else:
		plt.imshow(image)
	if c != 0 or r!= 0:
		plt.scatter([c],[r], color = 'red')
	if left_centroid[0] != -1 and left_centroid[1] != -1:
		plt.scatter([left_centroid[0]],[left_centroid[1]], color = 'red')
	if right_centroid[0] != -1 and right_centroid[1] != -1:
		plt.scatter([right_centroid[0]],[right_centroid[1]], color = 'red')
	plt.show()

#Displays multiple images on a signle matplotlib plot
#Returns nothing
#--------------------------------------------
#images - array of physical images
#grayscale - image is gray
def display_images(images, grayscale = True):
    num_images= len(images)
    columns = 4
    plt.figure(figsize=(20,10*num_images/4))
    for i, img in enumerate(images):
        plt.subplot(num_images // columns + 1, columns, i + 1)
        if grayscale:
        	plt.imshow(img, cmap='gray')
        else:
        	plt.imshow(img)
        plt.title(f'Image {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#Reads in points from a text file that are separated by commas
#Returns a 2xN or 3xN array of points
#---------------------------------------------
#filename - location/name.txt of file
def read_points_from_txt(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    points = [list(map(float, line.strip().split(','))) for line in lines]
    points_array = np.array(points).T
    return points_array

#Takes in an array of images and saves them to a folder
#Returns none
#------------------
#outer_path - location of outer folder
#inner_folder - location of inner folder
#images - array of images to save
#filename - text preceeding the number image
#extension - image type
def save_images(outer_path, inner_folder, images, filename = 'frame', extension = '.jpg'):
    full_path = os.path.join(outer_path, inner_folder)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    for i, image in enumerate(images):
        image_filename = f"{filename}{i}{extension}"
        image_path = os.path.join(full_path, image_filename)
        cv2.imwrite(image_path, image)

    print(f"Saved {len(images)} images to {full_path}")

#Takes in an image and a series of points and plots the points on the image
#Returns none
#------------------------
#image - physical image
#points - any points to be drawn on the image
#file_loc - folder the image is to be saved to
#filename - filename of image
#extension - image type
#grayscale - image is gray
#label - label the points or not
def overlay_points(image, points, file_loc = None, filename = None, extension = '.png', grayscale = True, label = False):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, point in enumerate(points):
        cv2.circle(image, (int(point[0]), int(point[1])), radius=10, color=(0, 0, 255), thickness=-1)
        if label:
            cv2.putText(image, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (255, 0, 0), 1, cv2.LINE_AA)

    if filename is not None:
        file_path = f'{file_loc}{filename}{extension}'
        cv2.imwrite(file_path, image)
    else:
        cv2.imshow("Overlay Points", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Write a file at file_name with the content provided
def write_file(file_name, content):
    with open(file_name, 'w') as file:
        file.write(content)