# convert data in letter.data to a stored numpy array
# containing only useful data
import csv
import numpy as np

def extract_data(x):
	return {
		'label': x[1],
		'pixels': x[6:134]
	}

pixels = []
labels = []

with open('letter.data') as file:
	reader = csv.reader(file, delimiter='\t')
	for row in reader:
		data = extract_data(row)
		pixels.append(data['pixels'])
		labels.append(data['label'])

np.save("pixels", pixels)
np.save("labels", labels)