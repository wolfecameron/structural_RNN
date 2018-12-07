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

	# translate([ 0,    0, 0]) rotate([0,0, $t*360/n1])                 color([1.00,0.75,0.75]) gear(mm_per_tooth,n1,thickness,hole);
	def get_SCAD_command(self, circular_pitch, gear_thick, hole_size, min_teeth):
		"""prints out an openSCAD command to create this gear
	
		circular pitch: gear constant for gears, defines the linear distance
		between the edges of teeth

		gear thick: the z axis thickness of gears

		hole size: the size of the hole in the center of the gear
		"""
		
		# check if gear needs to be hollow or not
		n_teeth = self.get_num_teeth(circular_pitch)
		if n_teeth < min_teeth:
			command = "gear"
		else:
			command = "hollow_gear"

		result = ""
		result += "translate([{0}, {1}, {2}])".format(str(self.pos[0]),
					str(self.pos[1]), str(self.pos[2]*gear_thick))
		result += "{0}({1}, {2}, {3}, {4});".format(command, str(circular_pitch),
					str(n_teeth), str(gear_thick),
					str(hole_size)) 
		return result	
	
	def __str__(self):
		"""to string method for gear object"""
		
		return "(r: {0}, pos: {1})".format(str(self.radius), str(self.pos))
