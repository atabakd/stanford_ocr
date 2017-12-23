# convert data in letter.data to a stored numpy array
# containing only useful data
import csv
import numpy as np
import string

def string_vectorizer(strng, alphabet=string.ascii_lowercase):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng]
    return vector[0]

def extract_data(x):
	return {
		'label': string_vectorizer(x[1]),
		'images': x[6:134]
	}

images = []
labels = []

with open('letter.data') as file:
	reader = csv.reader(file, delimiter='\t')
	for row in reader:
		data = extract_data(row)
		images.append(data['images'])
		labels.append(data['label'])

np.savez("letters", images=images, labels=labels)