"""Main deap evolution file for evolving the RNN to create different types of circular
structures - uses deap to evolve sets of weights that are inputted into RNN to yield output
"""

from circle_RNN import RNN
from deap_RNN_config import get_tb, N_IN, N_HID, N_OUT, N_GEN, RADIUS, MAX_POINTS. POP_SIZE
from deap_RNN_config import MUTPB, CXPB
from deap_RNN_help import list_to_matrices, inject_weights, get_rnn_output

# import toolbox from config file
toolbox = get_tb()

# instantiate the RNN that will be used to get fitnesses
rnn = RNN(N_IN, N_HID, N_OUT)

# instantiate the population
pop = toolbox.population()

# begin the evolutionary loop
for g in range(N_GEN):
	print("Running Generation {0}".format(str(g)))
	
	# get output for every individual in population and store in a list
	all_outputs = []
	for ind in pop:
		w1, w2 = list_to_matrices(ind, N_IN, N_HID, N_OUT)
		rnn = inject_weights(rnn, w1, w2)
		all_outputs.append(get_rnn_output(rnn, RADIUS, MAX_POINTS))  
	
	# get fitnesses from each of the outputs
	fits = toolbox.map(all_outputs, toolbox.evaluate)
	
	# assign fitness to individuals
	for ind. fit in zip(pop, fits):
		ind.fitness.values = fit
	
	# perform selection on the population to maximize fitness
	pop = toolbox.select(pop, k=POP_SIZE)

	# APPLY MUTATION AND CROSSOVER
	# both crossover and mutation are inplace operations
	for ind in pop:
		if np.random.uniform() <= MUTPB:
			toolbox.mutate(ind)
			del ind.fitness.values
	
	for child, child2 in zip(pop[::2], pop[1::2]):
		if np.random.uniform() <= CXPB:
			toolbox.mate(child1, child2)
			del mutant.fitness.values
