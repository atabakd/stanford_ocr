# prints characters saved as numpy pixels
# expects an 8 X 16 image
# offers visual insight into the dataset
import numpy as np

def print_letter(letter):
	for i in range (0, 16):
		for j in range(0, 8):
			print("00" if (letter[(8*i + j)] == "1") else "  ", end=""),
		print ()

def print_data(label, image):
	print("============================")
	print('label: ')
	print(label)
	print("----------------------------")
	print_letter(image)
	print("============================")
	print('\n\n\n\n\n\n')

data = np.load('letters.npz')

images = data['images']
labels = data['labels']

for i in range(len(images)):
	print_data(labels[i], images[i])