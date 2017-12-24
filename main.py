from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import string
import tensorflow as tf
import numpy as np

def extract_features(x):
	return {
		'letter': x[1],
		'pixels': x[6:134]
	}

def main(_):
	# filenames = tf.train.string_input_producer(["letter.data"])
	# reader = tf.TextLineReader()
	# key, value = reader.read(filenames)
	# all_cols = tf.decode_csv(
 #    value, record_defaults=[[""]]*135, field_delim="\t")

	data = np.load('letters.npz')

	images = data['images']
	labels = data['labels']
	# print(labels)

	# dataset = tf.data.Dataset.from_tensor_slices((images, labels))

	images_placeholder = tf.placeholder(images.dtype, images.shape)
	labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

	dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
	# iterator = dataset.make_initializable_iterator()
	# with tf.Session() as sess2:
	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(coord=coord)
		
	# 	for _ in range(200):
	# 		feature = sess2.run(all_cols)
	# 		extracted = extract_features(feature)
	# 		pixels.append(extracted['pixels'])
	# 		labels.append(string_vectorizer(extracted['letter']))
		
	# 	coord.request_stop()
	# 	coord.join()

	# dataset = tf.data.Dataset.from_tensor_slices((pixels, labels))
	# print(dataset)
	
	x = tf.placeholder(tf.float32, [None, 128])
	W = tf.Variable(tf.zeros([128, 26]))
	b = tf.Variable(tf.zeros([26]))
	y = tf.matmul(x, W) + b

	y_ = tf.placeholder(tf.float32, [None, 26])

	cross_entropy = tf.reduce_mean(
	    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# sess.run(iterator.initializer, feed_dict = {images_placeholder: images,
	# 											labels_placeholder: labels})
	
	# Train
	dataset = dataset.shuffle(buffer_size=10000)
	# dataset = dataset.batch(25)
	dataset = dataset.batch(500)
	# dataset = dataset.repeat(10)
	iterator = dataset.make_initializable_iterator()
	sess.run(iterator.initializer, feed_dict = {images_placeholder: images,
											labels_placeholder: labels})

	for i in range(500):
		# sess.run(iterator.initializer)
		data = sess.run(iterator.get_next())
		batch_labels = data[1]
		batch_examples = data[0]
		if i % 10 == 0:
			print(i)
			sess.run(iterator.initializer, feed_dict = {images_placeholder: images,
											labels_placeholder: labels})
			train_accuracy = accuracy.eval(feed_dict={
      	x: batch_examples, y_: batch_labels})
			print('step %d, training accuracy %g' % (i, train_accuracy))
		sess.run(train_step, feed_dict={x:	batch_examples, y_: batch_labels})

	# Test trained model
	print(sess.run(accuracy, feed_dict={x: images,
	                                    y_: labels}))

if __name__ == '__main__':
  tf.app.run(main=main, argv=[])