import numpy as np
import tensorflow as tf

from data_exporter import write_to_file
from tensorflow.contrib.layers import flatten

def write_variables():
	print "Writing to file"
	var = [v for v in tf.trainable_variables()]
	for v in var:
		print v
	for v in var:
		print v
		ev = v.eval()
		print ev
		write_to_file(ev, v.name)

		if v.name == 'test_weights/n_filters_conv1_w:0':
			print "NAME: ", v.name
			w,h,_,n = np.shape(ev)
			for i in xrange(0,n):
				print "\n\nFilter i: ",i
				print ev[:,:,:,i]
				print "FLAT: "
				flt = ev[:,:,:,i].flatten()
				for k in flt:
					print "%.3f " % k,
				print "\n"
def write_operation_result(var, prefix, variable_name):
	print "Result shape: ", np.shape(var), variable_name
	#print var

	print "Results: "
	shp = np.shape(var);
	if len(shp) >= 4:
		w,h,_,n = shp
		for i in xrange(0,n):
			print "\n\nResults for Filter i: ",i
			print var[:,:,:,i]
	else:
		print var
	write_to_file(var, prefix+variable_name)

def conv_test(number_of_filters, prefix):
	random_image_shape = (1, 6, 6, 1)
	random_image = tf.Variable(tf.truncated_normal(shape=random_image_shape, mean = 0, stddev = 0.1), name=prefix+"random_image")
	conv1_w = tf.Variable(tf.truncated_normal(shape = (3,3,1,number_of_filters), mean = 0, stddev = 0.1), name=prefix+'conv1_w')
	conv1 	= tf.nn.conv2d(random_image, conv1_w, strides = [1,1,1,1], padding = 'VALID', name=prefix+'conv1')
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		conv_result = sess.run(conv1)

		write_variables() # Ignoring the fact that are still previous variables in the session
		write_operation_result(conv_result, prefix, "conv1_result")
		#write_operation_result(maxpool_result, prefix, "maxpool_result")		

		print "Done!"

def maxpool_test(number_of_filters, prefix):
	random_image_shape = (1, 6, 6, 1)
	random_image = tf.Variable(tf.truncated_normal(shape=random_image_shape, mean = 0, stddev = 0.1), name=prefix+"random_image")
	maxpool = tf.nn.max_pool(random_image, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		maxpool_result = sess.run(maxpool)

		write_variables() # Ignoring the fact that are still previous variables in the session
		write_operation_result(maxpool_result, prefix, "maxpool_result")
		#write_operation_result(maxpool_result, prefix, "maxpool_result")		
	print "Maxpool Done!"


def matmul_test(number_of_filters, prefix):
	random_image_shape = (1, 6, 6, 1)
	random_image = tf.Variable(tf.truncated_normal(shape=random_image_shape, mean = 0, stddev = 0.1), name=prefix+"random_image")
	maxpool = tf.nn.max_pool(random_image, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
	
	fc0 = flatten(maxpool)
	
	fc3_w = tf.Variable(tf.truncated_normal(shape=(9,10), mean = 0, stddev = 0.1), name=prefix+'fc3_w')
	#fc3_b = tf.Variable(tf.zeros(10), name='fc3_b')
	print "fc0 shape: ", np.shape(fc0)
	print "fc3 shape: ", np.shape(fc3_w)
	logits = tf.matmul(fc0, fc3_w) # + fc3_b

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		logits = sess.run(logits)
		maxpool = sess.run(maxpool)

		write_variables() # Ignoring the fact that are still previous variables in the session
		write_operation_result(logits, prefix, "matmul_logits")
		write_operation_result(maxpool, prefix, "matmul_maxpool")
		#write_operation_result(maxpool_result, prefix, "maxpool_result")		
	print "matmul_test Done!"


def main():
	# Set random seed so we have consistent results
	tf.set_random_seed(0)

	#sh = (3,3,1,6)
	#test_variable = tf.Variable(tf.random_normal(shape = sh), name='test_var')
	#fill_ones_op = test_variable.assign(tf.ones(shape=sh))

	#another_shape=(3,3,6)
	#test_variable = tf.Variable(tf.random_normal(shape = another_shape), name='test_var_2')
	#ones_operation = test_variable.assign(tf.ones(shape=another_shape))
#
#	#with tf.Session() as sess:
#	#	sess.run(tf.global_variables_initializer())
#	#	sess.run(test_variable)
#
#	#	write_variables()
	#	print "Done!"


	#prefix = "convolution_op_test___"
	#print "####################################################################"
	#print "######################### %s ################### " % (prefix)
	#print "####################################################################"
	#conv_test(1, prefix)

	print "####################################################################"
	print "####################################################################"

	#conv_test(3, "test_weights/n_filters_")
	#maxpool_test(3, "test_weights/maxpool_test_")
	matmul_test(2, "test_weights/matmul_test_")
if __name__ == '__main__':
	main()