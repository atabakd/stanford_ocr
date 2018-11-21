# convert data in letter.data to a stored numpy array
# containing only useful data
from __future__ import division, print_function, absolute_import, unicode_literals
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
		'images': x[6:134],
    'fold': x[5],
		'next-id': x[2],
		'word-id': x[3],
	}


images = list()
labels = list()
folds = list()
nextIdxs = list()
wordIdxs = list()

with open('letter.data') as file:
  reader = csv.reader(file, delimiter=str('\t'))
  for row in reader:
    data = extract_data(row)
    images.append(data['images'])
    labels.append(data['label'])
    folds.append(data['fold'])
    nextIdxs.append(data['next-id'])
    wordIdxs.append(data['word-id'])

np.savez("letters", images=images, labels=labels, folds=folds, nextIdxs=nextIdxs, wordIdxs=wordIdxs)