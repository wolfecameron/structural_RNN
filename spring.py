"""this file implements a class for a torsional spring - stores all info about spring and can
retrieve needed info about the spring"""

import numpy as np

from deap_RNN_config import M_INIT, B_INIT, T_INIT, L_INIT
class Spring():

	def __init__(self, m=np.random.normal(M_INIT, .25, 1), b=np.random.normal(B_INIT, .25, 1), \
					t=np.random.normal(T_INIT, .25, 1), l=np.random.normal(1.0, .25, 1)):
		"""initializes the spring to values that are passed"""
		
		self.shape_slope = m[0]
		self.z_thick = b[0]
		self.thick = t[0]
		self.length = l[0]*L_INIT

	def get_k(self, modulus):
		"""returns the spring constant for the given torsional spring"""

		return (np.pi*self.z_thick*np.power(self.thick, 3.0)*modulus) \
				/(6.0*self.length)

	def get_torque(self, theta, modulus):
		"""given an angular displacement, this method returns the theoretical
		amount of torque the spring would output"""

		return self.get_k(modulus)*theta

	def get_params_as_list(self):
		"""creates a list of the torsional spring parameters of form
		[shape_slope, z_thick, thick, length] and returns it"""

		params = [self.shape_slope, self.z_thick, self.thick, self.length]
		return params
	
	def __str__(self):
		"""to string method for the Spring class"""
	
		result = ""
		result += "Torsional Spring Specs: \n"
		result += "Shape Eq. Slope: {0}\n".format(str(self.shape_slope))
		result += "Z Thickness: {0}\n".format(str(self.z_thick))
		result += "In-Plane Thickness: {0}\n".format(str(self.thick))
		result += "Spiral Length: {0}\n".format(str(self.length))

		return result
	
if __name__ == "__main__":
	"""use for simple testing"""

	s = Spring()
	print(s)
