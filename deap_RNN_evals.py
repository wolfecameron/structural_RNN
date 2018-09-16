"""contains all evaluation functions that are used in the deap ea for
the structural rnn
"""

import numpy as np

def specified_change_eval(position_list):
	"""evaluates positions in the list based on the closeness
	of each pair of adjacent points to a desired change in theta
	and r - tries to make every movement from one point to another
	match a certain change in theta and r
	"""
	
	'''
	# define target values for change in r and theta
	target_diff_r = 2.0
	target_diff_theta = .25
	
	# keep running total of the amount of difference from target
	total_diff = 0.0

	# go through each pair of adjacent elements
	# find value difference of theta and r
	for index in range(len(position_list) - 1):
		curr_pos = position_list[index]
		next_pos = position_list[index + 1]
		total_diff += np.square(target_diff_r - (curr_pos[0] - next_pos[0]))
		total_diff += np.square(target_diff_theta - (next_pos[1] - curr_pos[1]))

	
	return total_diff, 
	'''
	# define the target values
	target_r = 1.0
	target_theta = .05

	# find the total differences from the target values
	total_diff_theta = 0.0
	total_diff_r = 0.0
	for prev, nxt in zip(position_list[:], position_list[1:]):
		delta_theta = nxt[1] - prev[1]
		total_diff_theta += np.square(.1 - delta_theta)	
		delta_r = prev[0] - nxt[0]
		total_diff_r += np.square(1.0 - delta_r)
			
	
	return (total_diff_theta*total_diff_r, )
	
def changes_zerodist_eval(position_list):
	"""evaluates a torsional spring both based on minimizing
	the size of the changes in r and theta and reaching the
	origin before reaching the maximum number of points
	"""
	
	# track total differences in theta/r
	total_diff = 0.0
	total_thick = 0.0
	
	# go through each pair of points and find the change in r/theta
	for prev, nxt in zip(position_list[:], position_list[1:]):
		delta_theta = np.square(nxt[1] - prev[1])
		delta_r = np.square(prev[0] - nxt[0])
		# combine values of r and theta changes
		total_diff += (delta_theta + delta_r)
		total_thick += nxt[2]	

	# find the final position of the spring
	# want this to be as close to 0 as possible
	final_r = position_list[-1][0]
	
	# return product of the two things being minimized
	# two separate objectives not needed because there is no trade off
	return (total_diff*total_thick*final_r, )
