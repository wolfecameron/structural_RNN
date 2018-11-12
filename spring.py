"""this file implements a class for a torsional spring - stores all info about spring and can
retrieve needed info about the spring"""

import numpy as np

class Spring():

	def __init__(self, r0, m, b, t, l):
		"""initializes the spring to values that are passed"""
		
		self.shape_intercept = r0
		self.shape_slope = m
		self.z_thick = b
		self.thick = t
		self.length = l

	def get_k(self, modulus):
		"""returns the spring constant for the given torsional spring"""

		return (np.pi*self.z_thick*np.power(self.thick, 3.0)*modulus) \
				/(6.0*self.length)

	def get_torque(self, theta, modulus):
		"""given an angular displacement, this method returns the theoretical
		amount of torque the spring would output"""

		return self.get_k(modulus)*theta
