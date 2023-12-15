import cv2

from helper import *

#Creates a video from images
def create_video(image_folder, video_name, fps = 30, img_extension='.png'):
	images = read_images(image_folder, img_extension, grayscale = False, unsorted = True, convert = False)
	frame = images[0]
	height, width, layers = frame.shape
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
	for image in images:
		video.write(image)
	cv2.destroyAllWindows()
	video.release()

#Creates video for both kalman and arm tracking
def create_video_kalman_arms(person, stroke):
	image_folder = f'images/{person}/{stroke}/Tracked/Kalman'
	video_name = f'results/{person}/{stroke}/Kalman.mp4'
	create_video(image_folder, video_name)
	image_folder = f'images/{person}/{stroke}/Tracked/Arms'
	video_name = f'results/{person}/{stroke}/Arms.mp4'
	create_video(image_folder, video_name)

#SAMPLE USAGE
person = 'Sierra'
stroke = 'Free'
# tracking_type = 'Kalman'
# image_folder = f'images/{person}/{stroke}/Tracked/{tracking_type}'
# video_name = f'results/{person}/{stroke}/{tracking_type}.mp4'
# create_video(image_folder, video_name)
create_video_kalman_arms(person, stroke)