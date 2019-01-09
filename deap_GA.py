"""Implements the generation of gear mechanisms with variable length genome
GA - no RNN involved, used for comparison of effectiveness of RNN approach"""

from deap import base, tools, algorithms, creator

import numpy as np

# constants used for config
LEN_GENOME = 6
POP_SIZE = 50
WEIGHTS = (1.0,)

# list of all the possible gear sizes in mechanisms for GA
GEAR_RADII = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0]


# define function for creating an individual in the population
def create_ind():
	ind = creator.Individual()
	# append all sizes of gears to the individual
	for i in range(LEN_GENOME):
		ind.append(np.random.choice(GEAR_RADII))
	# append index for last gear in the system
	ind.append(np.random.randint(1, LEN_GENOME))
	return ind

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
# TODO: need a custom mutation operator
toolbox.register("select", tools.selTournament, tournsize=3)
# TODO: NEED EVALUATION
