# convert data in letter.data to a stored numpy array
# containing only useful data
import csv
import numpy as np

def extract_data(x):
	return {
		'label': x[1],
		'images': x[6:134]
	}

images = []
labels = []

with open('letter2.data') as file:
	reader = csv.reader(file, delimiter='\t')
	for row in reader:
		data = extract_data(row)
		images.append(data['images'])
		labels.append(data['label'])

np.savez("letters", images=images, labels=labels)