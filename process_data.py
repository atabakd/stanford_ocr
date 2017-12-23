import csv
import numpy as np

def extract_data(x):
	return {
		'label': x[1],
		'pixels': x[6:134]
	}

def print_letter(letter):
	for i in range (0, 16):
		for j in range(0, 8):
			print("0" if (letter[(8*i + j)] == "1") else " ", end=""),
		print ()

pixels = []
labels = []

with open('letter.data') as file:
	reader = csv.reader(file, delimiter='\t')
	for row in reader:
		data = extract_data(row)
		print_letter(data['pixels'])
		# print()
		# pixels.append(data['pixels'])
		# labels.append(data['label'])

np.save("pixels", pixels)