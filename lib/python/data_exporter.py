import numpy as np
import struct
import sys

unsigned_int_type = '<i'
float_type = '<f'
	
def _flat_and_write(f, file_name, arr):
	# Write flattened numbers
	arr = arr.flatten()
	#print arr
	print "Writing: ", np.shape(arr)
	print "float %s[] = {" % (file_name)
	for n in arr:
		print "%.3f, " % n,
		f.write(struct.pack(float_type, n))	
	print "}\n"

def write_to_file(arr, file_name):
	
	arr = arr.astype(dtype=np.float32)
	#print "ARR: ",arr
	print "Preparing creation of: ", file_name
	print "Loading: ", np.shape(arr)

	with open(file_name, 'wb') as f:
		# Write how many dimensions
		number_dimensions = len(np.shape(arr))
		f.write(struct.pack(unsigned_int_type, number_dimensions))
		print "# Number dimensions: ", number_dimensions

		# Write dimensions
		for dim in np.shape(arr):
			f.write(struct.pack(unsigned_int_type, dim))
			print "# Dim: ", dim

		shp = np.shape(arr)
		n_dim = len(shp)
		if n_dim <= 2:
			_flat_and_write(f, file_name, arr)
		elif n_dim == 4:
			# Assuming we are dealing with filters
			# @note there's probably a better way to handle this
			print "Using the fourth dimension"
			flat_truck = []
			for i in xrange(0, shp[3]):
				print "Dim: ", i
				_flat_and_write(f, file_name, arr[:,:,:,i])
		elif n_dim == 3:
			sys.exit('NOT IMPLEMENTED: SAVING ELEMENTS WITH 3 DIMENSIONS')
		else:
			sys.exit('NOT IMPLEMENTED: SAVING ELEMENTS WITH UNKNOWN DIMENSIONS')