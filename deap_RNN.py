"""Main deap evolution file for evolving the RNN to create different types of circular
structures - uses deap to evolve sets of weights that are inputted into RNN to yield output
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from circle_RNN import RNN
from deap_RNN_config import get_tb, N_IN, N_HID, N_OUT, N_GEN, MAX_POINTS, POP_SIZE, PLACEMENT_THRESH
from deap_RNN_config import MUTPB, CXPB, ACT_EXP, MAX_Y, MAX_X, MIN_GEARS, MAX_GEARS, STOP_THRESHOLD
from deap_RNN_config import RADIUS_SCALE, OUTPUT_MIN, X_BOUND, Y_BOUND, C_DICT
from deap_RNN_help import list_to_matrices, inject_weights, get_gear_ratio, create_mechanism_representation
from deap_RNN_help import get_mechanism_vector 
from deap_RNN_help import get_gear_mechanism as get_output
from vis_structs import vis_gears_nonlinear as vis_output

# import toolbox from config file
toolbox = get_tb()

# instantiate the population
pop = toolbox.population()

# begin the evolutionary loop
for g in range(N_GEN):
	print("Running Generation {0}".format(str(g)))
	
	# get output for every individual in population and store in a list
	all_outputs = []
	for ind in pop:
		# run the RNN to get gear mechanisms to evaluate
		rnn = RNN(N_IN, N_HID, N_OUT)
		w1, w1_bias, w2, w2_bias = list_to_matrices(ind, N_IN, N_HID, N_OUT)
		rnn = inject_weights(rnn, w1, w1_bias, w2, w2_bias)
		output = get_output(rnn, MAX_GEARS, MIN_GEARS, STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH)
		all_outputs.append(output)  
		
	# generate matrix of vectors for all individuals
	vec_list = []
	mechanism_list = []
	for ind in all_outputs:
		mechanism_list.append(create_mechanism_representation(ind, PLACEMENT_THRESH, OUTPUT_MIN))
		vec_list.append(get_mechanism_vector(mechanism_list[-1]))
	
	# stack all vectors together to create a matrix
	mech_matrix = np.vstack(vec_list)
	
	fits = []	
	# get average fit and append into running list
	for ind in mechanism_list:
		fits.append(toolbox.evaluate(ind, mech_matrix, X_BOUND, Y_BOUND))
	
	# assign fitness to individuals
	for ind, fit in zip(pop, fits):
		ind.fitness.values = fit
	
	# perform selection on the population to maximize fitness
	pop = toolbox.select(pop)
	
	# only apply mutation if not last generation
	if(g < N_GEN - 1):
		# APPLY MUTATION AND CROSSOVER
		# both crossover and mutation are inplace operations
		for ind in pop:
			if np.random.uniform() <= MUTPB:
				toolbox.mutate(ind)
				del ind.fitness.values
		
		for child1, child2 in zip(pop[::2], pop[1::2]):
			if np.random.uniform() <= CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values


# contains tuples of individuals and their associated fitness
# used to sort individual's output by fitness for viewing
ind_and_fits = []

mechanism_list = []
vec_list = []
# view results of the evolution
for count, ind in enumerate(pop):
	rnn = RNN(N_IN, N_HID, N_OUT)
	w1, w1_bias, w2, w2_bias = list_to_matrices(ind, N_IN, N_HID, N_OUT)
	rnn = inject_weights(rnn, w1, w1_bias, w2, w2_bias)
	# get output for each individual in final generation
	output_positions = get_output(rnn, MAX_GEARS, MIN_GEARS, STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH)
	# insert placeholder list into evaluation - only first fitness value matters for sorting
	mechanism_list.append(create_mechanism_representation(output_positions, PLACEMENT_THRESH, OUTPUT_MIN))
	vec_list.append(get_mechanism_vector(mechanism_list[-1]))

# stack all vectors into a matrix
mech_matrix = np.vstack(vec_list)

# go through all mechanisms and assign fitness
for mechanism in mechanism_list:
	fitness = toolbox.evaluate(mechanism, mech_matrix, X_BOUND, Y_BOUND)
	# append tuple of individual's outputs and fitness to the global list
	ind_and_fits.append((output_positions, fitness))

# sort list of outputs by fitness - only uses a single objective
ind_and_fits = sorted(ind_and_fits, key=lambda x: x[0], reverse=True)

# go through outputs sorted by fintess for viewing
for count, out in enumerate(ind_and_fits):
	vis_output(create_mechanism_representation(out[0], PLACEMENT_THRESH, OUTPUT_MIN), C_DICT)
	print(out[1])
	print("Now viewing individual {0}".format(str(count)))
