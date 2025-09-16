import cv2, os, random
import numpy as np
from glob import glob

def get_image_size():
	# Find the first available gesture image
	for gesture_folder in glob('gestures/*'):
		if os.path.isdir(gesture_folder):
			for image_file in glob(os.path.join(gesture_folder, '*.jpg')):
				img = cv2.imread(image_file, 0)
				if img is not None:
					return img.shape
	# Return a default or raise an error if no images are found
	raise FileNotFoundError("No gesture images found to determine size.")

gestures = os.listdir('gestures/')
gestures.sort(key = int)
begin_index = 0
end_index = 5
image_x, image_y = get_image_size()

if len(gestures)%5 != 0:
	rows = int(len(gestures)/5)+1
else:
	rows = int(len(gestures)/5)

full_img = None
for i in range(rows):
	col_img = None
	for j in range(begin_index, end_index):
		img_path = "gestures/%s/%d.jpg" % (j, random.randint(1, 1200))
		img = cv2.imread(img_path, 0)
		if np.any(img == None):
			img = np.zeros((image_y, image_x), dtype = np.uint8)
		if np.any(col_img == None):
			col_img = img
		else:
			col_img = np.hstack((col_img, img))

	begin_index += 5
	end_index += 5
	if np.any(full_img == None):
		full_img = col_img
	else:
		full_img = np.vstack((full_img, col_img))


cv2.imshow("gestures", full_img)
cv2.imwrite('full_img.jpg', full_img)
cv2.waitKey(0)
