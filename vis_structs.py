"""Creates visualizations of structures created by the circle RNN
using matplotlib"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def vis_spring(all_pts):
	"""takes in a list of tuples containing r and theta
	values and creates a polar plot of these coordinates
	with matplotlib
	"""
	
	# create a polar matplotlib axis
	axis = plt.subplot(111, projection="polar")
	
	# set title and graph the points	
	plt.title("Polar Graph of Torsional Spring")
	plt.plot([np.pi*(t[1] % 2.0) for t in all_pts], [t[0] for t in all_pts])
	plt.show()

def vis_spring_with_thickness(all_pts):
	"""takes in a list of tuples containing r, theta, and thickness
	and uses matplotlib to create a polar graph of the torsional
	spring including the thicknesses at each point
	"""
	
	# create polar axis for matplotlib
	axis = plt.subplot(111, projection='polar')
	
	# title the plot
	plt.title("Polar Graph of Torsional Spring")
	
	# go through each pair of adjacent points and add that
	# line for the two points into the graph
	for prev, nxt in zip(all_pts[:], all_pts[1:]):
		thetas = [np.pi*(prev[1] % 2.0), np.pi*(nxt[1] % 2.0)]
		radius = [prev[0], nxt[0]]
		plt.plot(thetas, radius, linewidth=2.0, color='black')
	
	# show the plot after all lines are added
	plt.show()


def vis_cartesian_output(all_pts):
	"""creates a visualization of an RNNs outputs in the cartesian
	space - these outputs are used to create the geometry for the
	tooth of a gear"""
	
	# create an axis to plot on
	axis = plt.subplot(111)
	
	# title the graph
	plt.title("Graph of Gear Tooth Geometry")
	
	# set equal limits of the axis
	plt.xlim(-.5, .5)
	plt.ylim(0.0, 1.0)

	# go through each pair of points in the geometry and graph them one by one
	for prev, nxt in zip(all_pts[:], all_pts[1:]):
		x_vals = [prev[0], nxt[0]]
		y_vals = [prev[1], nxt[1]]
		plt.plot(x_vals, y_vals, linewidth=2.0, color='black')

	plt.show()

def vis_gear_mechanism(outputs, pos_thresholds):
	"""takes all values from gear mechanism output of rnn and uses them
	to generate a visualization of the gear system with matplotlib, the
	pitch diameter of the gears is graphed

	each element of outputs is of form (radius, gear_pos, stop)

	pos_thresholds contains the two threshold values to interpret value of gear_pos
	if lower than two values, it is to left, if in middle it is attached to back, and
	if greater it is to right of the previous gear
	"""

	# create matplotlib axis to plot circles on
	fig, ax = plt.subplots()

	# instantiate variables to be used in plotting
	position = (0, 0)
	radius = outputs[0][0]
	circles = []
	all_pos = [0] # keep this so you know the maximum location
	
	# create circle object for first gear and add into list
	circles.append(plt.Circle(position, radius, alpha=.4))

	# go through all other outputs to create each gear
	for out_ind in range(1, len(outputs)):
		radius = outputs[out_ind][0]
		# direction dictates if gear placed to left, right, or attached to back
		# determined by gear_pos value in relation to position thresholds
		direction = 1
		if(outputs[out_ind][1] < pos_thresholds[0]):
			direction = -1
		elif(outputs[out_ind][1] >= pos_thresholds[0] and outputs[out_ind][1] <= pos_thresholds[1]):
			direction = 0
		
		# add up radius of current and previous gear to find change in x location for them to mesh
		pos_delta = outputs[out_ind][0] + outputs[out_ind - 1][0]
		position = (position[0] + direction*pos_delta, 0)
		all_pos.append(position[0])		

		# create and append circle object for current gear
		circles.append(plt.Circle(position, radius, alpha=.4))

	# plot all the circles on the matplotlib axis
	for c in circles:
		ax.add_artist(c)

    # set max/default window size of matplotlib
	x_max = max(all_pos)
	x_min = min(all_pos)
	all_rad = [x[0] for x in outputs]
	max_rad = 1.25*max(all_rad) # scale max radius up slightly so window not too tight
	ax.set_ylim((-max_rad, max_rad))
	ax.set_xlim((x_min - max_rad, x_max + max_rad)) 

	plt.show()

def vis_gears_nonlinear(mechanism, c_dict):
	"""takes in a list gear objects and outputs a matplotlib visualization
	of the gear system - where the gears may be placed at angles instead of
	always in a straight line"""

	# create matplotlib axis for the visualization
	fig, ax = plt.subplots()

	# create a circle object for each gear
	circles = []
	for gear in mechanism:
		circles.append(plt.Circle((gear.pos[0], gear.pos[1]), gear.radius, \
						alpha=.1, color=c_dict[gear.pos[2]]))

	# plot all circles onto the matplotlib axis
	for c in circles:
		ax.add_artist(c)
	
	# find bounds for creating the window of the visualization
	max_radius = max([x.radius for x in mechanism])
	max_x = max([abs(x.pos[0]) for x in mechanism])
	max_y = max([abs(x.pos[1]) for x in mechanism])
	x_lim = 1.1*(max_radius + max_x) # scale up a bit to give extra space
	y_lim = 1.1*(max_radius + max_y)
	ax.set_xlim((-x_lim, x_lim))
	ax.set_ylim((-y_lim, y_lim))

	plt.show()
	
if __name__ == '__main__':
	"""main loop used for simple testing"""
	
	output = [((0,0), 1.0), ((5, 2), 3.0), ((-5, -2), 1.5)]
	vis_gears_nonlinear(output)
