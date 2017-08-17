from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def save_fig(img, name, color="gray"):
	plt.imshow(img, cmap=color)
	plt.savefig(name+".png")

def load_dataset():

	mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
	x_train, y_train           = mnist.train.images, mnist.train.labels
	x_validation, y_validation = mnist.validation.images, mnist.validation.labels
	x_test, y_test             = mnist.test.images, mnist.test.labels


	print "##################################"
	print "############ dataset  ############"
	print "##################################"


	print "train"
	print np.shape(x_train)
	print np.shape(y_train)
	
	print "validation"
	print np.shape(x_validation)
	print np.shape(y_validation)
	
	print "test"
	print np.shape(x_test)
	print np.shape(y_test)
		

	print "##################################"
	print "##################################"

	sample_image = x_train[0,:,:,-1]
	print "x [0 sample] ", np.shape(sample_image)
	save_fig(sample_image, "training_sample")
	print "y", y_train
	print "y min: ", np.min(y_train)
	print "y max: ", np.max(y_train)
	print "y unique: ", np.unique(y_train, return_counts=True)

	print "##################################"
	print "##################################"

	# Do the shuffle
	x_train, y_train = shuffle(x_train, y_train)
	x_validation, y_validation = shuffle(x_validation, y_validation)
	x_test, y_test = shuffle(x_test, y_test)

	return x_train, y_train, x_validation, y_validation, x_test, y_test

def lenet(x):
	mu = 0
	sigma = 0.1
	# Architecture
	# Layer 1: Convolutional. In: 32x32x1 -> 28x28x6

	conv1_w = tf.Variable(tf.truncated_normal(shape = (5,5,1,6), mean = mu, stddev = sigma))
	conv1_b = tf.Variable(tf.zeros(6))
	conv1 	= tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1], padding = 'VALID', name='conv1') + conv1_b
	# Activation
	conv1   = tf.nn.relu(conv1) 
	# Pooling. In 28x28x6 -> 14x14x6
	conv1 	= tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')


	# Layer 2: Convolutional. In 14x14x6 -> 10x10x16
	conv2_w = tf.Variable(tf.truncated_normal(shape = (5,5,6,16), mean = mu, stddev = sigma))
	conv2_b = tf.Variable(tf.zeros(16))
	conv2 = tf.nn.conv2d(conv1, conv2_w, strides = [1,2,2,1], padding = 'VALID')

	# Fully connected In: 5x5x16 -> 400
	fc0 = flatten(conv2)

	# Layer 3 In: 400 -> 120
	fc1_w = tf.Variable(tf.truncated_normal(shape = (400, 120), mean = mu, stddev = sigma))
	fc1_b = tf.Variable(tf.zeros(120))
	fc1   = tf.matmul(fc0, fc1_w) + fc1_b

	# Activation
	fc1 = tf.nn.relu(fc1)

	# Layer 4
	fc2_w = tf.Variable(tf.truncated_normal(shape = (120, 84), mean = mu, stddev = sigma))
	fc2_b = tf.Variable(tf.zeros(84))
	fc2   = tf.matmul(fc1, fc2_w) + fc2_b

	# Activation
	fc2 = tf.nn.relu(fc2)

	# Layer 5 Fully Connected In: 84 -> 10
	fc3_w = tf.Variable(tf.truncated_normal(shape=(84,10), mean = mu, stddev = sigma))
	fc3_b = tf.Variable(tf.zeros(10))
	logits = tf.matmul(fc2, fc3_w) + fc3_b

	return logits

def evaluate(x_data, y_data, num_examples, batch_size, accuracy_operation, x, y):
	num_examples = len(x_data)
	total_accuracy = 0
	sess = tf.get_default_session()
	for offset in range(0, num_examples, batch_size):
		batch_x, batch_y = x_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
	
		accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
		total_accuracy += (accuracy * len(batch_x))
	return total_accuracy / num_examples

def pshape(s, data):
	print s, np.shape(data)

def main():
	x_train, y_train, x_validation, y_validation, x_test, y_test = load_dataset()

	# Pad images with 0s
	x_train      = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
	x_validation = np.pad(x_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
	x_test       = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

	print "After padding: ",  " x_train ", np.shape(x_train), " x_validation ", np.shape(x_validation), " x_test ", np.shape(x_test)
   


	epochs = 1
	batch_size = 128

	#
	# tf.placeholder with defines a tensor that is supposed to be fed through feed_dict
	#
	x = tf.placeholder(tf.float32, (None, 32,32,1))
	y = tf.placeholder(tf.int32, (None))

	#
	# tf.one_hot one hot encodes the expected result (3 ---> 0 0 0 1 0 0 0 0 0 0)
	#
	one_hot_y = tf.one_hot(y, 10)

	rate = 0.001

	logits = lenet(x)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate = rate)
	training_operation = optimizer.minimize(loss_operation)

	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()

	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		num_examples = len(x_train)
		print "Training!"

		for i in range(epochs):
			x_train, y_train = shuffle(x_train, y_train)
			for offset in range(0,num_examples, batch_size):
				end = offset + batch_size
				batch_x, batch_y = x_train[offset:end], y_train[offset:end]
				sess.run(training_operation, feed_dict = {x: batch_x, y: batch_y})

			validation_accuracy = evaluate(x_validation, y_validation, len(x_validation), batch_size, accuracy_operation, x,y)
			print "Epoch: ", i+1
			print "Validation accuracy ", validation_accuracy
			
		saver.save(sess,'lenet')
		print "Model saved!"

	with tf.Session() as sess:
		train_writer = tf.summary.FileWriter('./test', sess.graph)
		saver.restore(sess, tf.train.latest_checkpoint('.'))
		test_accuracy = evaluate(x_test, y_test, len(x_test), batch_size, accuracy_operation, x, y)
		print "Test accuracy ", test_accuracy


if __name__ == '__main__':
	main()