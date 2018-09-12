"""Main deap evolution file for evolving the RNN to create different types of circular
structures - uses deap to evolve sets of weights that are inputted into RNN to yield output
"""

import numpy as np

from circle_RNN import RNN
from deap_RNN_config import get_tb, N_IN, N_HID, N_OUT, N_GEN, RADIUS, MAX_POINTS, POP_SIZE
from deap_RNN_config import MUTPB, CXPB
from deap_RNN_help import list_to_matrices, inject_weights, get_rnn_output
from vis_structs import vis_coil

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
		rnn = RNN(N_IN, N_HID, N_OUT)
		w1, w1_bias, w2, w2_bias = list_to_matrices(ind, N_IN, N_HID, N_OUT)
		rnn = inject_weights(rnn, w1, w1_bias, w2, w2_bias)
		output = get_rnn_output(rnn, RADIUS, MAX_POINTS)
		all_outputs.append(output)  
	
	# get fitnesses from each of the outputs
	fits = []
	for out in all_outputs:
		fits.append(toolbox.evaluate(out))
	#fits = toolbox.map(all_outputs, toolbox.evaluate)
	
	# assign fitness to individuals
	for ind, fit in zip(pop, fits):
		ind.fitness.values = fit
	
	# perform selection on the population to maximize fitness
	pop = toolbox.select(pop, k=POP_SIZE)
	
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

# view results of the evolution
for count, ind in enumerate(pop):
	rnn = RNN(N_IN, N_HID, N_OUT)
	w1, w2 = list_to_matrices(ind, N_IN, N_HID, N_OUT)
	rnn = inject_weights(rnn, w1, w2)
	output_positions = get_rnn_output(rnn, RADIUS, MAX_POINTS)
	print(output_positions)
	vis_coil(output_positions)
	print("Now viewing individual {0}".format(str(count)))
