"""Main deap evolution file for evolving the RNN to create different types of gear
train mechanisms- uses deap to evolve sets of weights that are inputted into RNN to yield output
"""

from copy import deepcopy
import pickle
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from circle_RNN import RNN
from gear import Gear
from deap_RNN_config import get_tb, N_IN, N_HID, N_OUT, N_GEN, POP_SIZE, PLACEMENT_THRESH
from deap_RNN_config import MUTPB, CXPB, ACT_EXP, MAX_Y, MAX_X, MIN_GEARS, MAX_GEARS, STOP_THRESHOLD
from deap_RNN_config import RADIUS_SCALE, OUTPUT_MIN, X_BOUND, Y_BOUND, C_DICT, GEAR_RADII
from deap_RNN_config import CIRCULAR_PITCH, GEAR_THICKNESS, HOLE_SIZE, NUM_UNIQUE_GEARS
from deap_RNN_config import POP_FILE, VEC_FILE, ARCH_FILE, MECH_FILE, GEAR_DISTS, HOLE_R, SLOT_LEN, DIST_FROM_CENT
from deap_RNN_config import INIT_OFFSET, SLOT_HT, SLOT_T
from deap_RNN_help import list_to_matrices, inject_weights, get_gear_ratio, create_discrete_mechanism
from deap_RNN_help import get_mechanism_vector, get_mech_and_vec, gen_openSCAD_beams 
#def gen_openSCAD_beams(mech, gear_dists, hole_r, slot_len, dist_from_cent):		
from deap_RNN_evalg import apply_mutation, apply_crossover
from deap_RNN_help import get_discrete_gear_mechanism as get_output
from vis_structs import vis_gears_nonlinear as vis_output

# import toolbox from config file
toolbox = get_tb()

# instantiate the population
pop = toolbox.population()

# all RNNs have the same number of hidden nodes set in config
for p in pop:
	p.h_nodes = N_HID

# this list holds the most novel individual from each generation
# surrogate finds fitness for each of these individuals
ARCHIVE = []
ARCHIVE_MATRIX = None
# begin the evolutionary loop
for g in range(N_GEN):
	print(f'Running Generation {g}')


	all_outputs = []
	vec_list = []
	mechanism_list = []	
	# get output for every individual in population and store in a list
	for ind in pop:
		rnn = RNN(N_IN, ind.h_nodes, N_OUT)
		output, mech, vec = get_mech_and_vec(ind, rnn, N_IN, N_OUT, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, \
				STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, GEAR_RADII, OUTPUT_MIN) 
		all_outputs.append(output)
		mechanism_list.append(mech)
		vec_list.append(vec)
	
	"""
	all_outputs = []
	for ind in pop:
		# run the RNN to get gear mechanisms to evaluate
		rnn = RNN(N_IN, ind.h_nodes, N_OUT)
		w1, w1_bias, w2, w2_bias = list_to_matrices(ind, N_IN, ind.h_nodes, N_OUT)
		rnn = inject_weights(rnn, w1, w1_bias, w2, w2_bias)
		output = get_output(rnn, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, STOP_THRESHOLD, \
						RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, 'one')
		all_outputs.append(output)
	
	# generate matrix of vectors for all individuals
	vec_list = []
	mechanism_list = []
	for ind in all_outputs:
		mechanism_list.append(create_discrete_mechanism(ind, GEAR_RADII, PLACEMENT_THRESH, OUTPUT_MIN))
		#print(ind)
		#vis_output(mechanism_list[-1], C_DICT)
		vec_list.append(get_mechanism_vector(mechanism_list[-1]))	
	"""

	# stack all vectors together to create a matrix
	mech_matrix = np.vstack(vec_list)
	# normalize the mechanism matrix by simply dividing by avg column value
	col_avg = np.mean(mech_matrix, axis=0) + .001	
		
	fits = []
	total_bound_CV = []
	total_intersect_CV = []
	total_axis_CV = []
	total_gear_CV = []	
	# get average fit and append into running list
	for ind, mech in zip(pop, mechanism_list):
		# create vector for individual and normalize it
		mech_vec = get_mechanism_vector(mech)/col_avg
		
		#print(mech_vec)
		#input()	
		# only evaluate based on pop in the first gen
		# normalize current vec and archive with col_avg
		if(g == 0):
			# use k == 3 to disclude distance from self in mech matrix
			fit_tup = toolbox.evaluate(mech, mech_vec, mech_matrix/col_avg, X_BOUND, Y_BOUND, HOLE_SIZE,k=3)
		# evaluate based on archive in other gens
		else:
			# can use k == 1 because self should not be in the archive
			fit_tup = toolbox.evaluate(mech, mech_vec, ARCHIVE_MATRIX/col_avg, X_BOUND, Y_BOUND, HOLE_SIZE)
		fits.append(fit_tup)
			
		# only consider nonzero terms in normalization to avoid watering down CV
		# many of the CV values will be 0 and would throw off the average
		if(fit_tup[1] > 0.0): 
			total_bound_CV.append(fit_tup[1])
		if(fit_tup[2] > 0.0):
			total_intersect_CV.append(fit_tup[2])
		if(fit_tup[3] > 0.0):
			total_axis_CV.append(fit_tup[3])
		if(fit_tup[4] > 0.0):
			total_gear_CV.append(fit_tup[4])

	# convert cv lists to numpy arrays
	total_bound_CV = np.array(total_bound_CV)
	total_intersect_CV = np.array(total_intersect_CV)
	total_axis_CV = np.array(total_axis_CV)
	total_gear_CV = np.array(total_gear_CV)

	# assign fitness and CV to individuals
	for ind, fit in zip(pop, fits):
		ind.fitness.values = fit[0],
		# CV should be the normalized sum of the constraint types
		# must ensure any divide by zero is avoided
		bound_cv = 0.0 if total_bound_CV.shape[0] == 0 else \
				(fit[1]/(np.mean(total_bound_CV)))
		intersect_cv = 0.0 if total_intersect_CV.shape[0] == 0 else \
				(fit[2]/(np.mean(total_intersect_CV)))
		axis_cv = 0.0 if total_axis_CV.shape[0] == 0 else \
				(fit[3]/(np.mean(total_axis_CV)))
		gear_cv = 0.0 if total_gear_CV.shape[0] == 0 else \
				(fit[4]/(np.mean(total_gear_CV)))
		ind.CV = bound_cv + intersect_cv + axis_cv + gear_cv
	
	# sort individuals and handle CV values
	valid_pop = [i for i in pop if i.CV <= 0.0]
	invalid_pop = [i for i in pop if i.CV > 0.0]
	lowest_valid = min(valid_pop, key=lambda x: x.fitness.values[0]).fitness.values[0]
	# change all fitnesses of invalid pop to be lower than lowest valid fit
	# subtract CV from the lowest valid fitness - creates gradient even for invalid
	for i in invalid_pop:
		i.fitness.values = (lowest_valid - i.CV,)	

	valid_pop.extend(invalid_pop)
	pop = valid_pop
	
	# BOOK KEEPING FOR ARCHIVE
	# append the most novel individual into the archive, update matrix with its vector
	best_ind = max(pop, key=lambda x: x.fitness.values[0])
	ARCHIVE.append(deepcopy(best_ind))
	# get all output information for next archive ind
	rnn = RNN(N_IN, ARCHIVE[-1].h_nodes, N_OUT)
	arch_out, arch_mech, arch_vec = get_mech_and_vec(ARCHIVE[-1], rnn, N_IN, N_OUT, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, \
			STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, GEAR_RADII, OUTPUT_MIN) 
	
	"""
	w1, w1_bias, w2, w2_bias = list_to_matrices(ARCHIVE[-1], N_IN, ARCHIVE[-1].h_nodes, N_OUT)
	rnn = inject_weights(rnn, w1, w1_bias, w2, w2_bias)
	arch_out = get_output(rnn, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, 'one')
	# update archive matrix from current vector
	arch_mech = create_discrete_mechanism(arch_out, GEAR_RADII, PLACEMENT_THRESH, OUTPUT_MIN)
	arch_vec = get_mechanism_vector(arch_mech)
	#print(best_ind.fitness.values[0])
	#vis_output(arch_mech, C_DICT)
	"""
	if(g == 0):
		ARCHIVE_MATRIX = np.vstack([arch_vec])
	else:
		ARCHIVE_MATRIX = np.vstack([ARCHIVE_MATRIX, arch_vec])

	# perform selection on the population to maximize fitness
	pop = toolbox.select(pop, k=len(pop))
	
	# only apply mutation if not last generation
	if(g < N_GEN - 1):
		# APPLY MUTATION AND CROSSOVER
		# both crossover and mutation are inplace operations
		apply_mutation(pop, toolbox, MUTPB)
		apply_crossover(pop, toolbox, CXPB, N_IN, N_OUT)		
		"""
		for ind in pop:
			if np.random.uniform() <= MUTPB:
				toolbox.mutate(ind)
				del ind.fitness.values
		
		for child1, child2 in zip(pop[::2], pop[1::2]):
			# find number of hidden nodes in each individual
			one_n = child1.h_nodes
			two_n = child2.h_nodes
			
			# find cutoff for hidden/output weights
			one_hid = (one_n + N_IN)*one_n + one_n # (num_hid + n_in)*n_hid + n_hid 
			one_end = one_hid + one_n*N_OUT + N_OUT # hidden weights + n_hid*n_out + n_out
			two_hid = (two_n + N_IN)*two_n + two_n
			two_end = two_hid + two_n*N_OUT + N_OUT
			rand = np.random.uniform()
			
			# 50-50 chance of using either crossover operator
			if rand <= (CXPB/2.0):
				child1[ :one_hid], child2[ :two_hid] = toolbox.ins_mate(child1[ :one_hid], child2[ :two_hid])
				child1[one_hid: one_end], child2[two_hid: two_end] = toolbox.ins_mate(child1[one_hid: one_end], child2[two_hid: two_end])
				del child1.fitness.values
				del child2.fitness.values
			elif (CXPB/2.0) < rand <= CXPB:
				child1[ :one_hid], child2[ :two_hid] = toolbox.ex_mate(child1[ :one_hid], child2[ :two_hid], CXPB)
				child1[one_hid: one_end], child2[two_hid: two_end] = toolbox.ex_mate(child1[one_hid: one_end], child2[two_hid: two_end], CXPB)
				del child1.fitness.values
				del child2.fitness.values
		"""

# write archive vectors to csv file
# write instructions to construct mechanisms to file
with open(ARCH_FILE, "w") as f:
	arch_vecs = []
	for ind in ARCHIVE:
		# find vector and mechanism representation
		rnn = RNN(N_IN, ind.h_nodes, N_OUT)
		output, mech, vec = get_mech_and_vec(ind, rnn, N_IN, N_OUT, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, \
				STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, GEAR_RADII, OUTPUT_MIN) 
		arch_vecs.append(vec)
		#mech = [Gear(28.0, (28.0, 28.0, 0), 0), Gear(8.0, (64.0, 28.0, 0), 0)]
		#mech = [Gear(12.0, (12.0, 12.0, 0), 0), Gear(24.0, (48.0, 12.0, 0), 0), Gear(8.0, (80.0, 12.0, 0), 1)]
		beams = gen_openSCAD_beams(mech, GEAR_DISTS, HOLE_R, SLOT_LEN, SLOT_HT, SLOT_T, DIST_FROM_CENT, INIT_OFFSET)
		print(beams)
		#def gen_openSCAD_beams(mech, gear_dists, hole_r, slot_len, dist_from_cent):		
		vis_output(mech, C_DICT)		
		# write mechanism info for printing to a separate file
		counter = 0
		while(os.path.isfile(MECH_FILE + str(counter) + ".txt")):
			counter += 1
		with open((MECH_FILE + str(counter) + ".txt"), "w") as mech_f:
			for g in mech:
				mech_f.write(str(g))
				mech_f.write("\n")
	# write all archive vectors into file	
	writer = csv.writer(f)
	writer.writerows(arch_vecs)	


# generate vectors for the population and write to csv file
with open(VEC_FILE, "w") as f:
	pop_vecs = []
	for ind in pop:
		rnn = RNN(N_IN, ind.h_nodes, N_OUT)
		output, mech, vec = get_mech_and_vec(ind, rnn, N_IN, N_OUT, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, \
				STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, GEAR_RADII, OUTPUT_MIN) 
		pop_vecs.append(vec)
	# write vector contents into csv file
	writer = csv.writer(f)
	writer.writerows(vec_list)
	

# pickle the population to be read during next evolution
with open(POP_FILE, "wb") as f:	
	pickle.dump(pop, f)

"""
# contains tuples of individuals and their associated fitness
# used to sort individual's output by fitness for viewing
ind_and_fits = []
outs = []
mechanism_list = []
vec_list = []
# view results of the evolution
for count, ind in enumerate(ARCHIVE):
	rnn = RNN(N_IN, ind.h_nodes, N_OUT)
	w1, w1_bias, w2, w2_bias = list_to_matrices(ind, N_IN, ind.h_nodes, N_OUT)
	rnn = inject_weights(rnn, w1, w1_bias, w2, w2_bias)
	# get output for each individual in final generation
	output_positions = get_output(rnn, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, \
			STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, 'one')
	# insert placeholder list into evaluation - only first fitness value matters for sorting
	outs.append(output_positions)
	mechanism_list.append(create_discrete_mechanisa(output_positions, GEAR_RADII, PLACEMENT_THRESH, OUTPUT_MIN))
	#vec_list.append(get_mechanism_vector(mechanism_list[-1]))

plt.title("Size of Mechanisms in Archive")
plt.hist([len(g) for g in mechanism_list], bins=(MAX_GEARS-MIN_GEARS))
plt.show()
"""
"""
# stack all vectors into a matrix and normalize
mech_matrix = np.vstack(vec_list)
col_avg = np.mean(mech_matrix, axis=0) + .001
mech_matrix /= col_avg


# go through all mechanisms and assign fitness
for ind, mechanism in zip(ARCHIVE, mechanism_list):
	# get individual vector and normalize
	vis_output(mechanism, C_DICT)
"""
