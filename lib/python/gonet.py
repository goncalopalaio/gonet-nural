from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from data_exporter import write_to_file


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
def gonet(x):
	mu = 0
	sigma = 0.1

	conv1_w = tf.Variable(tf.truncated_normal(shape = (3,3,1,6), mean = mu, stddev = sigma), name='conv1_w')
	conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
	#op_conv2d_plus_b 	= tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1], padding = 'VALID', name='conv1') + conv1_b
	op_conv2d 	= tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1], padding = 'VALID', name='conv1')
	op_conv2d_plus_b = op_conv2d + conv1_b
	print "CONV: ", op_conv2d_plus_b
	
	# Activation
	op_relu   = tf.nn.relu(op_conv2d_plus_b) 
	op_maxpool = tf.nn.max_pool(op_relu, ksize = [1,5,5,1], strides = [1,3,3,1], padding = 'VALID')
	print "CONV_FINAL:", op_maxpool

	fc0 = flatten(op_maxpool)
	#fc0 = tf.nn.relu(fc0)
	print fc0

	fc3_w = tf.Variable(tf.truncated_normal(shape=(384,10), mean = mu, stddev = sigma), name='fc3_w')
	fc3_b = tf.Variable(tf.zeros(10), name='fc3_b')
	
	op_matmul = tf.matmul(fc0, fc3_w)
	op_resmatmul_plus_b = op_matmul + fc3_b 
	logits = op_resmatmul_plus_b

	return logits, op_matmul, op_resmatmul_plus_b, op_conv2d,op_conv2d_plus_b ,op_relu ,op_maxpool, fc0

def output_arr(arr, out_file):
	a = arr.astype(dtype=np.float16)
	with open(out_file, 'wb') as f:
		l = len(a.tobytes()) / 2
		f.write(struct.pack('I', l))
		f.write(a.tobytes())


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


	epochs = 1
	batch_size = 128

	#
	# tf.placeholder with defines a tensor that is supposed to be fed through feed_dict
	#
	x = tf.placeholder(tf.float32, (None, 28,28,1))
	y = tf.placeholder(tf.int32, (None))

	#
	# tf.one_hot one hot encodes the expected result (3 ---> 0 0 0 1 0 0 0 0 0 0)
	#
	one_hot_y = tf.one_hot(y, 10)

	rate = 0.001

	logits, op_matmul, op_resmatmul_plus_b, op_conv2d,op_conv2d_plus_b ,op_relu ,op_maxpool, fc0 = gonet(x)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate = rate)
	training_operation = optimizer.minimize(loss_operation)

	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()

	prefix_filename = "gonet_weights/gonet_"
	
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
		var = [v for v in tf.trainable_variables()]
		print "VARS: \n",var
		for v in var:
			print v.name
			write_to_file(v.eval(), prefix_filename + v.name)
		print "Model saved!"

	with tf.Session() as sess:
		train_writer = tf.summary.FileWriter('./test', sess.graph)
		saver.restore(sess, tf.train.latest_checkpoint('.'))
		test_accuracy = evaluate(x_test, y_test, len(x_test), batch_size, accuracy_operation, x, y)
		print "Test accuracy ", test_accuracy


	# Write example image
	index_to_sample = 0
	print "x_test: ", np.shape(x_test)
	x_sample = x_test[index_to_sample,:,:,-1]
	print "x_sample: ", np.shape(x_sample)
	write_to_file(x_sample, prefix_filename + "x_test_image_sample")

	print "y_test: ", np.shape(y_test)
	y_sample = np.array(y_test[index_to_sample])
	print "y_sample: ", np.shape(y_sample)
	write_to_file(y_sample, prefix_filename + "y_test_image_sample")


	# Save a few numbers for later
	write_to_file(x_sample, prefix_filename + "x_test_image_sample_number_" + str(y_sample))


	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('.'))
		
		x_sample = x_test[np.newaxis,index_to_sample,:,:,:] # todo no need to index from x_test
		logits, op_matmul, op_resmatmul_plus_b, op_conv2d,op_conv2d_plus_b ,op_relu ,op_maxpool, fc0 = sess.run([logits, op_matmul, op_resmatmul_plus_b, op_conv2d,op_conv2d_plus_b ,op_relu ,op_maxpool, fc0], feed_dict = {x: x_sample , y: y_sample})

		for name, res in [("op_matmul",op_matmul), ("op_resmatmul_plus_b",op_resmatmul_plus_b), ("op_conv2d", op_conv2d),("op_conv2d_plus_b",op_conv2d_plus_b) ,("op_relu", op_relu) ,("op_maxpool", op_maxpool), ("fc0", fc0), ("logits", logits)]:
			print "op: %s : shape: %s : vals: %s \n" % (name, np.shape(res), res)
			
			write_to_file(res, prefix_filename + name+"_op_dump")


if __name__ == '__main__':
	main()