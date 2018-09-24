"""contains all evaluation functions that are used in the deap ea for
the structural rnn
"""

import sys

import numpy as np

#from deap_RNN_help import get_cartesian_coordinates

def specified_change_eval(position_list):
	"""evaluates positions in the list based on the closeness
	of each pair of adjacent points to a desired change in theta
	and r - tries to make every movement from one point to another
	match a certain change in theta and r
	"""

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

def loops_eval(position_list):
	"""evaluates RNN based on the number of loops that
	are created in its output divided by the distance
	of last point to 0 - this should be maximized
	"""

	# keep tracks of the number of loops in output
	num_loops = 0.0

	# go through each adjacent pair of positions and check for loop
	for prev, nxt in zip(position_list[:], position_list[1:]):
		# there is a loop if modulus of theta decreased - passed by 2
		if(nxt[1] % 2.0 < prev[1] % 2.0):
			num_loops += 1

	# get the the last radius position of the output
	# must add a small value to it to avoid divide by 0
	final_r = position_list[-1][0] + .01 

	return (num_loops/final_r, )

def distance_between_lines_eval(position_list):
	"""evaluates RNN output based on the closest point in
	the spiral to each line formed by adjacent points in
	the spiral - tries to avoid lines within the RNN from
	colliding with each other by maximizing the distance from
	each line segment to surrounding points
	"""

	# convert everything to cartesian coordinates	
	cartesian_coords = []

	# go through each position and convert to cartesian
	for tup in position_list:
		r = tup[0]
		theta = tup[1]
		x = r*np.cos(np.pi*theta)
		y = r*np.sin(np.pi*theta)

		# append each cartesian as a tuple into the resulting list
		cartesian_coords.append((x,y))
	
	# find minimum distance between the line from each pair of points
	# and another point in the torsional spring - keep a running total
	total_dist = 0.0	
	for prev, nxt in zip(cartesian_coords[:], cartesian_coords[1:]):
		x1, y1 = prev
		x2, y2 = nxt
		
		# go through each point and find least distance
		min_dist = sys.maxsize
		for point in cartesian_coords:
			x0, y0 = point
			numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
			denom = np.sqrt(np.square(y2 - y1) + np.square(x2 - x1))
			distance = numerator/denom
			# keep track of shortest distance
			if(distance < min_dist):
				min_dist = distance
		# increment total distance with minimum distance for each
		# pair of points within the spring
		total_dist += min_dist
	
	return total_dist,
