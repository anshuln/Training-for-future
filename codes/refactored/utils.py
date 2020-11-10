'''Utils.py

Misc utility functions
''' 
import numpy as np

def get_cumulative_data_indices(source_data_indices):
	
	'''Returns cumulative indices
	
	for a list of lists [[x1,x2...],[y1,y2,..],[z1,z2,z3,...]] returns  [[x1,x2,..],[x1,x2,...,y1,y2,...],[x1,x2,..,y1,y2,..,z1,z2,z3...]]
	
	Arguments:
		source_data_indices {[type]} -- list of lists
	
	''' 
	curr_indices = source_data_indices[0].tolist()
	# if curr_indices
	cumulative_indices = [[x for x in curr_indices]]
	for i in range(1,len(source_data_indices)):
		curr_indices = curr_indices + source_data_indices[i].tolist()
		cumulative_indices.append([x for x in curr_indices])
	return cumulative_indices

def store_numpy_array(filepath, array, allow_pickle=True, fix_imports=True):

	with open(filepath, 'wb') as file:
		np.save(file, array, allow_pickle, fix_imports)
