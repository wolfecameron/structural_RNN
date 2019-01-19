"""implements the second phase of evolution that directly evolves based on distance achieved in physical tests

entire file only runs one generation of the evolution so that evolution can be controlled by the
matlab surrogate code
"""

import csv
import os

import pickle
import numpy as np

from circle_RNN import RNN
from deap_RNN_help import get_mech_and_vec
from deap_RNN_config import get_tb, MUTPB_DIST, N_IN, N_OUT, CXPB_DIST, HOLE_SIZE
from deap_RNN_config import FIT_FILE, POP_FILE, VEC_FILE, ARCH_FILE, MECH_FILE
from deap_RNN_config import NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, STOP_THRESHOLD
from deap_RNN_config import RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, GEAR_RADII, OUTPUT_MIN
from deap_RNN_config import X_BOUND, Y_BOUND, HOLE_SIZE, GEAR_DISTS,  HOLE_R, SLOT_LEN, SLOT_HT
from deap_RNN_config import SLOT_T, DIST_FROM_CENT, INIT_OFFSET, SLOT_HOLE_LEN, SLOT_HOLE_HT
from deap_RNN_help import check_bounding_box, check_intersect_amount, check_conflicting_gear_axis
from deap_RNN_help import eval_useless_gears, gen_openSCAD_beams
from deap_RNN_evalg import apply_mutation, apply_crossover

# initialize the deap tb
tb = get_tb()

# set seed number in numpy for reproducing results
seed_f = open("seed.txt", "r")
np.random.seed(int(seed_f.readlines()[0]))
seed_f.close()

# read in smaller pop from pickle file
f = open(POP_FILE, "rb")
archive = pickle.load(f)
f.close()

# assign all the archive fitnesses to the archive
with open(FIT_FILE, "r") as f:
	fits = f.readlines()
	for fit, ind in zip(fits, archive):
		fit = float(fit)
		ind.fitness.values = fit,

# take 6 individuals from archive to form smaller population
# take best 3 individuals, 2 from upper remaining half, 1 from lower remaining half
pop = []
archive = sorted(archive, key=lambda x: x.fitness.values, reverse=True)
pop.extend(archive[:3])
index_one = np.random.randint(3, 12)
pop.append(archive[index_one])
index_two = np.random.randint(3, 12)
while(index_two == index_one):
	index_two = np.random.randint(3, 12)
pop.append(archive[index_two])
index_three = np.random.randint(12, 20)
pop.append(archive[index_three])


# write fitnesses from archive into the fitness file to begin evolution
with open(FIT_FILE, "w") as f:
	for ind in pop:
		f.write(str(ind.fitness.values[0]))
		f.write("\n")

# evolve smaller pop directly for 5 generation
for g in range(5):
	# read in all fitnesses for population
	with open(FIT_FILE, "r") as f:
		fits = f.readlines()
		for fit, ind in zip(fits, pop):
			fit = float(fit)
			print(fit)
			ind.fitness.values = fit,
	input()
	# use to find averages of all CV values
	fits = []
	bound_CV = []
	intersect_CV = []
	axis_CV = []
	gear_CV = []
	# determine CV for each individual 
	for ind in pop:
		rnn = RNN(N_IN, ind.h_nodes, N_OUT)
		output, mech, vec = get_mech_and_vec(ind, rnn, N_IN, N_OUT, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, \
				STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, GEAR_RADII, OUTPUT_MIN)	
	
		# find all different CV on mechanism
		CV_bound = check_bounding_box(mech, X_BOUND, Y_BOUND)
		CV_intersect = check_intersect_amount(mech)
		CV_axis = check_conflicting_gear_axis(mech, HOLE_SIZE)
		CV_gear = eval_useless_gears(mech)	
		fits.append((ind.fitness.values[0], CV_bound, CV_intersect, CV_axis, CV_gear))

		# append CV values into list to find averages
		if(CV_bound > 0.0):
			bound_CV.append(CV_bound)
		if(CV_intersect > 0.0): 
			intersect_CV.append(CV_intersect)
		if(CV_axis > 0.0):
			axis_CV.append(CV_axis)
		if(CV_gear > 0.0):
			gear_CV.append(CV_gear)

	# convert CV lists to numpy to find averages
	bound_CV = np.array(bound_CV)
	intersect_CV = np.array(intersect_CV)
	axis_CV = np.array(axis_CV)
	gear_CV = np.array(gear_CV)

	# normalize CV values for each ind
	for ind, fit in zip(pop, fits):
		ind.fitness.values = fit[0],
		
		# CV should be the normalized sum of the constraint types
		# must ensure any divide by zero is avoided
		total_bound_cv = 0.0 if bound_CV.shape[0] == 0 else \
				(fit[1]/(np.mean(bound_CV)))
		total_intersect_cv = 0.0 if intersect_CV.shape[0] == 0 else \
				(fit[2]/(np.mean(intersect_CV)))
		total_axis_cv = 0.0 if axis_CV.shape[0] == 0 else \
				(fit[3]/(np.mean(axis_CV)))
		total_gear_cv = 0.0 if gear_CV.shape[0] == 0 else \
				(fit[4]/(np.mean(gear_CV)))
		ind.CV = total_bound_cv + total_intersect_cv + total_axis_cv + total_gear_cv
	
	# separate into valid and invalid individuals	
	valid_pop = []
	invalid_pop = []
	for ind in pop:
		if(ind.CV > 0.0):
			invalid_pop.append(ind)
		else:
			valid_pop.append(ind)

	# find minimum valid fitness
	min_fit = min(valid_pop, key=lambda x: x.fitness.values[0]).fitness.values[0]

	# assign all invalid individuals below the minimum fitness
	for ind in invalid_pop:
		ind.fitness.values = (min_fit - ind.CV),

	# combine valid and invalid pops that now have correct fitness
	valid_pop.extend(invalid_pop)
	pop = valid_pop 

	# perform selection on the population to maximize fitness
	offspring = tb.select(pop, k=len(pop))
	# clone offspring
	offspring = list(tb.map(tb.clone, offspring))

	# apply mutation and perform crossover
	apply_mutation(offspring, tb, MUTPB_DIST)
	apply_crossover(offspring, tb, CXPB_DIST, N_IN, N_OUT)

	# set population equal to result of mutation and crossover
	pop[:] = offspring

	# write information for new pop into files for testing 
	for i, ind in enumerate(pop):
		# get mechanism representation	
		rnn = RNN(N_IN, ind.h_nodes, N_OUT)
		output, mech, vec = get_mech_and_vec(ind, rnn, N_IN, N_OUT, NUM_UNIQUE_GEARS, MAX_GEARS, MIN_GEARS, \
				STOP_THRESHOLD, RADIUS_SCALE, ACT_EXP, PLACEMENT_THRESH, GEAR_RADII, OUTPUT_MIN)	
	
		# check if individual is invalid
		CV_bound = check_bounding_box(mech, X_BOUND, Y_BOUND)
		CV_intersect = check_intersect_amount(mech)
		CV_axis = check_conflicting_gear_axis(mech, HOLE_SIZE)
		CV_gear = eval_useless_gears(mech)	
	
		# generate the openSCAD commands for the mechanism inserts	
		beams = gen_openSCAD_beams(mech, GEAR_DISTS, HOLE_R, SLOT_LEN, SLOT_HT, SLOT_T, \
				DIST_FROM_CENT, INIT_OFFSET, SLOT_HOLE_LEN, SLOT_HOLE_HT)
		
		# write info for every gear to the file
		with open((MECH_FILE + str(i) + ".txt"), "w") as f:
			if(CV_bound > 0 or CV_intersect > 0 or CV_axis > 0 or CV_gear > 0):
				f.write("***** MECHANISM INVALID *****\n")
			for g in mech:
				f.write(str(g))
				f.write("\n")
			f.write("\n\nopenSCAD script:\n")
			f.write(beams)

	input(f"Populate the fit file with {len(pop)} fitnesses, then press enter.")
