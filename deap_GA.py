"""Implements the generation of gear mechanisms with variable length genome
GA - no RNN involved, used for comparison of effectiveness of RNN approach"""

from deap import base, tools, algorithms, creator

import numpy as np

from deap_RNN_evals import phase_one_eval
from deap_RNN_config import X_BOUND, Y_BOUND, HOLE_SIZE


# constants used for config
LEN_GENOME = 6
POP_SIZE = 50
WEIGHTS = (1.0,)
N_GEN = 20

# list of all the possible gear sizes in mechanisms for GA
GEAR_RADII = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0]


def create_ind():
	"""creates a single individual in the population"""

	ind = creator.Individual()
	# append all sizes of gears to the individual
	for i in range(LEN_GENOME):
		gear_rad = np.random.choice(GEAR_RADII)
		coax = np.random.choice([0, 1, 2])
		ind.append((gear_rad, coax))
	# append index for last gear in the system
	ind.append(np.random.randint(1, LEN_GENOME + 1))
	return ind

def mutate_ind(ind, mutpb):
	"""mutates an individual in the population"""

	for i in range(LEN_GENOME):
		# check if this gear in genome should be mutated
		new_rad = ind[i][0]
		new_coax = ind[i][1]
		if(np.random.uniform() <= mutpb):
			new_rad = np.random.choice(GEAR_RADII)
		if(np.random.uniform() <= mutpb):
			new_coax = np.random.choice([0, 1, 2])
		new_gear = (new_rad, new_coax)
		ind[i] = new_gear

	
	# check if length of mechanism should be mutated
	if(np.random.uniform() <= mutpb):
		new_len = np.random.randint(1, LEN_GENOME + 1)
		ind[LEN_GENOME] = new_len
		

# DEAP CONFIG
# create types needed for deap
creator.create("FitnessMulti", base.Fitness, weights=WEIGHTS)
creator.create("Individual", list, fitness=creator.FitnessMulti, CV=0.0, h_nodes=-1)

# initialize the toolbox
toolbox = base.Toolbox()

# register function to create individual in the toolbox
toolbox.register("create_ind", create_ind)
toolbox.register("individual", toolbox.create_ind)


# register function to create population in the toolbox
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

# register all functions needed for evolution in the toolbox
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_ind)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", phase_one_eval)

# evolutionary loop/initialization
pop = toolbox.population()
for i in range(N_GEN):
	print(f"Running Generation {i}")

	


def phase_one_eval(mech, mech_vec, other_vecs, x_bound,dd y_bound, hole_size, k=1):
