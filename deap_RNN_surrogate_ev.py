"""implements the second phase of evolution utilizing the surrogate to assign fitnesses to individuals
within the population

entire file only runs one generation of the evolution so that evolution can be controlled by the
matlab surrogate code
"""

import pickle
import csv

from deap_RNN_config import get_tb, MUTPB, N_IN, N_OUT, CXPB
from deap_RNN_config import FIT_FILE, POP_FILE, VEC_FILE
from deap_RNN_evalg import apply_mutation, apply_crossover

# initialize the deap toolbox
toolbox = get_tb()

# read pop in from pickle file
f = open(POP_FILE, "r")
pop = pickle.load(f)
f.close()

# read in all fitnesses for population
with open(FIT_FILE, "r") as f:
	fits = f.readlines()
	for fit, ind in zip(fits, pop):
		fit = float(fit)
		ind.fitness.values = fit,
"""
CV_bound = check_bounding_box(mech, x_bound, y_bound)
CV_intersect = check_intersect_amount(mech) 
CV_axis = check_conflicting_gear_axis(mech, hole_size)
"""

# TODO: assess CV to change fitness
# select population based on fitness from surrogate
pop = tb.select(pop)

# apply mutation and perform crossover
apply_mutation(pop, tb, MUTPB)
apply_crossover(pop, tb, CXPB, N_IN, N_OUT)
	

# generate vector for each individual in resulting pop
# write all vectors to vec file
with open(VEC_FILE, "w") as f:
	vec_list = []
	for ind in pop:
		rnn = RNN(N_IN, ind.h_nodes, N_OUT)
		w1, w1_bias, w2, w2_bias = list_to_matrices(ind, N_IN, ind.h_nodes, N_OUT)
		rnn = inject_weights(rnn, w1, w1_bias, w2, w2_bias)
		out = get_output(rnn, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, \
				STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, 'one')
		mech = create_discrete_mechanism(out, GEAR_RADII, PLACEMENT_THRESH, OUTPUT_MIN)
		vec = list(get_mechanism_vector(mech))
		vec_list.append(arch_vec)
	# write vector contents into csv file
	writer = csv.writer(f)
	writer.writerows(vec_list)

# write population into a file with pickle
f = open(POP_FILE, "wb")
pop = pickle.dump(pop, f)
f.close()
