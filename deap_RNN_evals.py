"""contains all evaluation functions that are used in the deap ea for
the structural rnn
"""

def specified_change_eval(position_list):
	"""evaluates positions in the list based on the closeness
	of each pair of adjacent points to a desired change in theta
	and r - tries to make every movement from one point to another
	match a certain change in theta and r
	"""
	
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
		total_diff += abs(target_diff_r - abs(curr_pos[0] - next_pos[0]))
		total_diff += abs(target_diff_theta - abs(curr_pos[1] - next_pos[1]))
	
	return total_diff, 
	
		 
