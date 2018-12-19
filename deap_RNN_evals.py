"""contains all evaluation functions that are used in the deap ea for
the structural rnn
"""

import sys

import numpy as np

from deap_RNN_help import get_gear_ratio, get_centers_and_radii, check_intersect
from deap_RNN_help import create_mechanism_representation, get_mechanism_vector
from deap_RNN_help import find_novelty, check_intersect_amount, check_bounding_box
from deap_RNN_help import check_conflicting_gear_axis

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

	return num_loops,#(num_loops/final_r, )

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

def loops_and_novelty_eval(positions_list, all_positions):
	"""evaluates spring on two metrics, number of loops
	and the average distance of this positions list to all others
	in the population"""

	# first find result of loop evaluation
	loop_val = loops_eval(positions_list)[0]

	# declare a sample size of points within each coord list to check
	sample_size = 5	

	# find average distance of current positions list to all others
	total_distance = 0.0
	for other_pos in all_positions:
		# pick 5 points in each of the structures to check similarity
		for i in range(sample_size):
			rand_index = sys.maxsize
			while(not (rand_index < len(positions_list) and rand_index < len(other_pos))):
					rand_index = int(np.random.uniform()*len(positions_list))
			coord = positions_list[rand_index]
			other_coord = other_pos[rand_index]
			# add radius distance to total distance
			total_distance += np.square(coord[0] - other_coord[0])
			# add theta distance to total distance
			total_distance += np.square(coord[1] - other_coord[1])
	# divide the total_distance by the total number of items_checked
	total_distance /= len(all_positions)
	
	# return two fitness metrics in a tuple
	return (loop_val, total_distance)

def gear_tooth_eval(positions_list, all_pos):
	"""evaluates the gear tooth cartesian output for RNN based on how far
	out the tooth goes in the x direction and making it back to 0 once it
	reaches the top position in y"""

	# find the final distance from 0 x
	final_x_dist = np.square(positions_list[-1][0])

	# get all x coordinates in numpy array so mean can be easily calculated
	all_x = [np.square(x[0]) for x in positions_list]
	x_np = np.array(all_x)

	total_distance = 0.0
	# evaluate novelty - based on distance of outputs to other outputs in group
	for other_out in all_pos:
		for pos, o_pos in zip(positions_list, other_out):
			total_distance += np.sqrt(np.square(pos[0] - o_pos[0]) + np.square(pos[1] - o_pos[1]))

	# make total distance an average by dividing by number of other outputs
	total_distance /= len(all_pos) 
			

	return ((final_x_dist)*np.mean(x_np), total_distance)

def gear_mechanism_eval(outputs):
	""" basic evaluation method for the gear mechanisms, just takes into accout
	the size of the mechanism and the differences in radius """

	# get number of gears and a list of radius
	radii = np.array([x[0] for x in outputs])
	placements = np.array([x[1] for x in outputs])
	
	return ((np.var(radii)*np.var(placements)*len(outputs)), )
	
def gear_mechanism_novelty_eval(outputs, all_outs, pos_thresh):
	""" basic evaluation method for the gear mechanisms, just takes into accout
	the size of the mechanism and the differences in radius """

	# get fitness from normal evaluation
	fitness = get_gear_ratio(outputs, pos_thresh)

	# find the difference between this solution and all other solutions
	total_diff = 0.0
	for o in all_outs:
		for curr, other in zip(outputs, o):
			total_diff += np.square(curr[0] - other[0])
			total_diff += np.square(curr[1] - other[1])
		
	return (fitness, total_diff)

def eval_nonlin_gears(outputs, spring, placement_thresh, output_min):
	"""performs simple evaluation on set of gears by checking if they
	intersect and how many gears are in system"""
	
	# get mechanism representation from the outputs
	mechanism = create_mechanism_representation(outputs, placement_thresh, output_min)	
	
	# find all gears that are an output in the system
	outputs = []
	for g in mechanism:
		if len(g.next_gears) == 0:
			outputs.append(g)	

	# find torque of each gear to get the average 
	total_torque = 0.0
	input_torque = spring.get_torque(2.0*np.pi, 1.0)
	for g in outputs:
		total_torque += (input_torque/g.ratio)
	total_torque /= len(outputs)
	
	return total_torque,

def phase_one_eval(mech, mech_vec, other_vecs, x_bound, y_bound, hole_size):
	"""evaluates an individual for phase one of the experiment, which
	selects individuals based on novelty and viability. Novelty is evaluated
	based on a characteristic vector desribing all properties of a mechanism
	while viability is determine based on intersecting gears

	ind vec and other vecs expected to both be normalized before they are
	passed into this equation

	return: novelty of the current vector and two values for amount of constraint violation
	"""	

	# maximize the gear ratio and novelty
	nov = find_novelty(mech_vec, other_vecs)
	
	# find the amount of constraint violation
	CV_bound = check_bounding_box(mech, x_bound, y_bound)
	CV_intersect = check_intersect_amount(mech) 
	CV_axis = check_conflicting_gear_axis(mech, hole_size)
	
	return (nov, CV_bound, CV_intersect, CV_axis)
