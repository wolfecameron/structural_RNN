"""implements a testing of numerous different topologies of RNN, the lower bounds and upper bounds of nodes
that are tested are listed in the configuration file, creates N examples of output for each topology to be
examined and determine the optimal number of hidden nodes"""

from circle_RNN import RNN
from deap_RNN_config import get_tb, MIN_NODES, MAX_NODES, N_IN, N_OUT
from deap_RNN_config import MIN_GEARS, MAX_GEARS, STOP_THRESHOLD, RADIUS_SCALE
from deap_RNN_config import ACT_EXP, PLACEMENT_THRESH, C_DICT, OUTPUT_MIN
from deap_RNN_help import get_gear_mechanism as get_output
from deap_RNN_help import list_to_matrices, inject_weights, create_mechanism_representation
from vis_structs import vis_gears_nonlinear as vis_output

# instantiate deap toolbox to create populations of weights
tb = get_tb()

# loop through each possible number of nodes to examine output
for n in range(MIN_NODES, MAX_NODES + 1):
	print("\n\nBEGINNING NEW TOPOLOGY")
	print("Hidden Nodes: {0}\n\n".format(str(n)))

	# create candidate list of weights to be examined
	# all weight matrices are same length, but only a portion used with smaller topology
	pop = tb.population()

	# go through each randomly generated individual and view the output
	for i, ind in enumerate(pop):
		# insert all current weight parameters into RNN
		rnn = RNN(N_IN, n, N_OUT)
		w1, w1_bias, w2, w2_bias = list_to_matrices(ind, N_IN, n, N_OUT)
		rnn = inject_weights(rnn, w1, w1_bias, w2, w2_bias)

		# obtain and view output
		output = get_output(rnn, MAX_GEARS, MIN_GEARS, STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, 'one')
		m = create_mechanism_representation(output, PLACEMENT_THRESH, OUTPUT_MIN)
		print("Viewing individual {0} with {1} hidden nodes".format(str(i), str(n)))
		vis_output(m, C_DICT)
