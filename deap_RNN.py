"""Main deap evolution file for evolving the RNN to create different types of circular
structures - uses deap to evolve sets of weights that are inputted into RNN to yield output
"""

from circle_RNN import RNN
from deap_RNN_config import get_tb, N_IN, N_HID, N_OUT, N_GEN, RADIUS, MAX_POINTS
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
	
	for ind in pop:
		w1, w2 = list_to_matrices(ind, N_IN, N_HID, N_OUT)
		rnn = inject_weights(rnn, w1, w2)
		print(get_rnn_output(rnn, RADIUS, MAX_POINTS))  
		input()

