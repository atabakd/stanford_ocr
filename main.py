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

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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

	W_conv1 = weight_variable([4, 4, 1, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1, 16, 8, 1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([4 * 2 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 2*4*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 26])
	b_fc2 = bias_variable([26])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	y_ = tf.placeholder(tf.float32, [None, 26])

	cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
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

	for i in range(5000):
		# sess.run(iterator.initializer)
		data = sess.run(iterator.get_next())
		batch_labels = data[1]
		batch_examples = data[0]
		if i % 100 == 0:
			print(i)
			sess.run(iterator.initializer, feed_dict = {images_placeholder: images,
											labels_placeholder: labels})
			train_accuracy = accuracy.eval(feed_dict={
      	x: batch_examples, y_: batch_labels, keep_prob: 1.0})
			print('step %d, training accuracy %g' % (i, train_accuracy))
		sess.run(train_step, feed_dict={x:	batch_examples, y_: batch_labels, keep_prob: 0.5})

	# Test trained model
	print(sess.run(accuracy, feed_dict={x: images,
	                                    y_: labels, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run(main=main, argv=[])