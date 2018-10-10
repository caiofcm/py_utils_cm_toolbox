import h5py
import numpy as np

############################################
###########################################
#
#
# 	 DICTIONARY TO HDF5
#
#
###########################################
###########################################


def save_dict_to_hdf5(dic, filename):
	with h5py.File(filename, 'w') as h5file:
		recursively_save_dict_contents_to_group(h5file, '/', dic)
	pass

def save_dict_to_open_file(dic, fstream):
	recursively_save_dict_contents_to_group(fstream, '/', dic)
	pass


def load_dict_from_hdf5(filename):
	with h5py.File(filename, 'r') as h5file:
		return recursively_load_dict_contents_from_group(h5file, '/')
	pass


def load_dict_from_hdf5_listing(filename):
	with h5py.File(filename, 'r') as h5file:
		return recursively_load_dict_contents_from_group_listing(h5file, '/')
	pass


def recursively_save_dict_contents_to_group(h5file, path, dic,
			opts = {}):

	if 'np_single_column' not in opts:
		opts['np_single_column'] = False
	if 'remove_zero_size' not in opts:
		opts['remove_zero_size'] = False

	# argument type checking
	if not isinstance(dic, dict):
		raise ValueError("must provide a dictionary")

	if not isinstance(path, str):
		raise ValueError("path must be a string")
	if not isinstance(h5file, h5py._hl.files.File):
		raise ValueError("must be an open h5py file")
	# save items to the hdf5 file
	for key, item in dic.items():
		#print(key,item)
		key = str(key)
		if isinstance(item, list):
			item = np.array(item)
			#print(item)
		if not isinstance(key, str):
			raise ValueError("dict keys must be strings to save to hdf5")
		# save strings, numpy.int64, and numpy.float64 types
		if isinstance(item, (np.int64, np.float64, str, np.float, float, np.float32, int)):
			#print( 'here' )
			h5file[path + key] = item
			if not h5file[path + key].value == item:
				raise ValueError(
					'The data representation in the HDF5 file does not match the original dict.')
		# save numpy arrays
		elif isinstance(item, np.ndarray):
			if len(item) == 0:
				if opts['remove_zero_size']:
					pass
				else:
					h5file[path + key] = item
			else:
				if opts['np_single_column']:
					if isinstance(item[0], np.ndarray) and item[0].size > 0:
						for j, vl in enumerate(item):
							recursively_save_dict_contents_to_group(h5file,
													path + key + '/', {str(j): vl})
					else:
						try:
							h5file[path + key] = item
						except:
							item = np.array(item).astype('|S9')
							h5file[path + key] = item
				else: #CONNECT TO FORMER IF (JUST FOR REFEREEENCE)
						try:
							h5file[path + key] = item
						except:
							item = np.array(item).astype('|S9')
							h5file[path + key] = item
				# if not np.array_equal(h5file[path + key].value, item):
				# 	raise ValueError('The data representation in the HDF5 file does not match the original dict.')
		# save dictionaries
		elif isinstance(item, dict):
			recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
		elif isinstance(item, np.bytes_):
			h5file[path + key] = item			
		# other types cannot be saved and will result in an error
		else:
			#print(item)
			raise ValueError('Cannot save %s type.' % type(item))

	pass


def recursively_load_dict_contents_from_group(h5file, path):

	ans = {}
	for key, item in h5file[path].items():
		if isinstance(item, h5py._hl.dataset.Dataset):
			ans[key] = item.value
		elif isinstance(item, h5py._hl.group.Group):
			ans[key] = recursively_load_dict_contents_from_group(
				h5file, path + key + '/')
	return ans


def recursively_load_dict_contents_from_group_listing(h5file, path):

	ans = {}
	for key, item in h5file[path].items():
		if isinstance(item, h5py._hl.dataset.Dataset):
			ans[key] = item.value.tolist()
		elif isinstance(item, h5py._hl.group.Group):
			ans[key] = recursively_load_dict_contents_from_group_listing(
				h5file, path + key + '/')
	return ans
