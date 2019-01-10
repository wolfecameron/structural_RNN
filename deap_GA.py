"""Implements the generation of gear mechanisms with variable length genome
GA - no RNN involved, used for comparison of effectiveness of RNN approach"""

import os

from deap import base, tools, algorithms, creator
import numpy as np
import pickle

from deap_GA_config import get_tb, LEN_GENOME, POP_SIZE, WEIGHTS, N_GEN, MUTPB, CXPB, GEAR_RADII
from deap_RNN_evals import phase_one_eval
from deap_RNN_config import X_BOUND, Y_BOUND, HOLE_SIZE, GEAR_DISTS, HOLE_R, SLOT_LEN, SLOT_HT, SLOT_T
from deap_RNN_config import DIST_FROM_CENT, INIT_OFFSET, SLOT_HOLE_LEN, SLOT_HOLE_HT
from deap_RNN_config import POP_FILE, VEC_FILE, ARCH_FILE, MECH_FILE
from deap_GA_help import mechanism_from_GA
from deap_RNN_help import get_mechanism_vector

# retrieve toolbox from config file
toolbox = get_tb()

# store all info for archive
ARCHIVE = []
ARCHIVE_MATRIX = []

# evolutionary loop/initialization
pop = toolbox.population()
for i in range(N_GEN):
	print(f"Running Generation {i}")

	# get vector and mechanism for each individual
	vecs = []
	mechs = []
	for ind in pop:
		mech = mechanism_from_GA(ind)
		vec = get_mechanism_vector(mech)
		mechs.append(mech)
		vecs.append(vec)
	
	# stack all vectors together to create a matrix
	mech_matrix = np.vstack(vecs)
	# normalize the mechanism matrix by simply dividing by avg column value
	col_avg = np.mean(mech_matrix, axis=0) + .001	
		
	fits = []
	total_bound_CV = []
	total_intersect_CV = []
	total_axis_CV = []
	total_gear_CV = []	
	# get average fit and append into running list
	for ind, mech in zip(pop, mechs):
		# create vector for individual and normalize it
		mech_vec = get_mechanism_vector(mech)/col_avg
		
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
	
	# create mech and vec for the best individual
	arch_mech = mechanism_from_GA(best_ind)	
	arch_vec = get_mechanism_vector(arch_mech)

	if(g == 0):
		ARCHIVE_MATRIX = np.vstack([arch_vec])
	else:
		ARCHIVE_MATRIX = np.vstack([ARCHIVE_MATRIX, arch_vec])

	# perform selection on the population to maximize fitness
	offspring = toolbox.select(pop, k=len(pop))
	# clone offspring
	offspring = toolbox.map(toolbox.clone, offspring)

	# perform mutation and crossover
	if(i < N_GEN - 1):
		# MUTATION
		for ind in offspring:
			if np.random.uniform() <= MUTPB:
				toolbox.mutate(ind)
				del ind.fitness.values	

		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if np.random.uniform() <= CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values
	
	pop[:] = offspring

# write archive vectors to csv file
# write instructions to construct mechanisms to file
with open(ARCH_FILE, "w") as f:
	arch_vecs = []
	for ind in ARCHIVE:
		# find vector and mechanism representation
		mech = mechanism_from_GA(ind)
		vec = get_mechanism_vector(mech)
		arch_vecs.append(vec)
		beams = gen_openSCAD_beams(mech, GEAR_DISTS, HOLE_R, SLOT_LEN, SLOT_HT, SLOT_T, DIST_FROM_CENT, INIT_OFFSET, SLOT_HOLE_LEN, SLOT_HOLE_HT)
		vis_output(mech, C_DICT)		
		# write mechanism info for printing to a separate file
		counter = 0
		while(os.path.isfile(MECH_FILE + str(counter) + ".txt")):
			counter += 1
		with open((MECH_FILE + str(counter) + ".txt"), "w") as mech_f:
			for g in mech:
				mech_f.write(str(g))
				mech_f.write("\n")
			mech_f.write("\n")
			mech_f.write(beams)
	# write all archive vectors into file	
	writer = csv.writer(f)
	writer.writerows(arch_vecs)	


# generate vectors for the population and write to csv file
with open(VEC_FILE, "w") as f:
	pop_vecs = []
	for ind in pop:
		mech = mechanism_from_GA(ind)
		vec = get_mechanism_vector(mech)
		pop_vecs.append(vec)
	# write vector contents into csv file
	writer = csv.writer(f)
	writer.writerows(vec_list)
	

# pickle the population to be read during next evolution
with open(POP_FILE, "wb") as f:	
	pickle.dump(pop, f)
