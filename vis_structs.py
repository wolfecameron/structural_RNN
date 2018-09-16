"""Creates visualizations of structures created by the circle RNN
using matplotlib"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def vis_coil(all_pts):
	"""takes in a list of tuples containing r and theta
	values and creates a polar plot of these coordinates
	with matplotlib
	"""
	
	# create a polar matplotlib axis
	axis = plt.subplot(111, projection="polar")
	
	# set title and graph the points	
	plt.title("Polar Graph of Torsional Spring")
	plt.plot([np.pi*(t[1] % 2.0) for t in all_pts], [t[0] for t in all_pts], linewidth=2.5)
	plt.show()

if __name__ == '__main__':
	"""Main conditional block for simple testing"""

	points = []
	vis_circle(points)


	 
