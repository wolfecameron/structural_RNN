"""contains all helper functions used in the pure GA evolution"""

from gear import Gear

def mechanism_from_GA(ind):
	"""create mechanism representation from GA genome"""	

	r = ind[0][0]
	mechanism = [Gear(r, (r, r, 0), 0)]
	end_ind = ind[-1]
	for index, curr in enumerate(ind[1: end_ind]):
		prev_gear = mechanism[-1]

		# get position for the next gear
		new_pos = (prev_gear.pos[0] + prev_gear.radius + curr[0], prev_gear.pos[1], prev_gear.pos[2])
		if(curr[1] == 2):
			new_pos = (prev_gear.pos[0], prev_gear.pos[1], prev_gear.pos[2] + 1)
		elif(curr[1] == 0):
			new_pos = (prev_gear.pos[0], prev_gear.pos[1], prev_gear.pos[2] - 1)
		mechanism.append(Gear(curr[0], new_pos, len(mechanism) - 1))
		prev_gear.next_gears.append(index + 1)
	
	return mechanism	

