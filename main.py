from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import string

import tensorflow as tf

def extract_features(x):
	return {
		'letter': x[1],
		'pixels': x[6:134]
	}
	
def string_vectorizer(strng, alphabet=string.ascii_lowercase):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng.decode("utf-8")]
    return vector[0]

def main(_):
	filenames = tf.train.string_input_producer(["letter.data"])
	reader = tf.TextLineReader()
	key, value = reader.read(filenames)
	all_cols = tf.decode_csv(
    value, record_defaults=[[""]]*135, field_delim="\t")

	pixels = []
	labels = []

	with tf.Session() as sess2:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		for _ in range(200):
			feature = sess2.run(all_cols)
			extracted = extract_features(feature)
			pixels.append(extracted['pixels'])
			labels.append(string_vectorizer(extracted['letter']))
		
		coord.request_stop()
		coord.join()

	dataset = tf.data.Dataset.from_tensor_slices((pixels, labels))
	print(dataset)
	
	x = tf.placeholder(tf.float32, [None, 128])
	W = tf.Variable(tf.zeros([128, 26]))
	b = tf.Variable(tf.zeros([26]))
	y = tf.matmul(x, W) + b

	y_ = tf.placeholder(tf.float32, [None, 26])

	cross_entropy = tf.reduce_mean(
	    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	
	# Train
	for _ in range(200):
	  sess.run(train_step, feed_dict={x: pixels, y_: labels})
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	# with tf.Session() as sess2:
	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(coord=coord)
		
		# for i in range(50):
		# 	feature = sess2.run(all_cols)
		# 	extracted = extract_features(feature)
		# 	pixels.append(extracted['pixels'])
		# 	labels.append(string_vectorizer(extracted['letter']))
		# 	if(i%100 == 0):
		# 		print (i)
		
		# coord.request_stop()
		# coord.join()

	# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: pixels,
	                                    y_: labels}))


if __name__ == '__main__':
  tf.app.run(main=main, argv=[])