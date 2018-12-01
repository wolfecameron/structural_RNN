"""this file implements a class for a gear that stores all information needed for a gear within a mechanism"""

import numpy as np

class Gear():

	def __init__(self, rad, pos, prev_gear):
		"""define all properties of the gear class, only radius, position, and
		prev gear are instantiated by the constructor
		"""
		
		self.radius = rad
		self.pos = pos # of form (x, y, z)
		self.prev_gear = prev_gear # stored as an index
		self.next_gears = [] # stored as a list of indices
		self.ratio = 1.0 # stored as a float
	
	def get_num_teeth(self, circular_pitch):
		"""determines the number of teeth that should be used for
		the current gear"""

		# calculate number of teeth in the gear and round to nearest int
		n_teeth = (np.pi*2.0*self.radius)/circular_pitch
		return round(n_teeth)	

	def __str__(self):
		"""to string method for gear object"""
		
		return "(r: {0}, pos: {1})".format(str(self.radius), str(self.pos))
