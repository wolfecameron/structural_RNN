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

if __name__ == '__main__':
	"""Main conditional block for simple testing"""

	points = []
	vis_circle(points)


	 
