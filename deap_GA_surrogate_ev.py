"""implements evolution past the 20th generation for pure GA experiment"""

import os
import csv

import numpy as np

from deap_GA_config import get_tb
from deap_RNN_config import X_BOUND, Y_BOUND, HOLE_SIZE, 
from deap_RNN_config import FIT_FILE, POP_FILE, VEC_FILE, ARCH_FILE, MECH_FILE
from deap_RNN_help import check_bounding_box, check_intersect_amount, mechanism_from_GA, get_mechanism_vector
from deap_RNN_help import check_conflicting_gear_axis, check_useless_gears
import pickle

# initialize the deap toolbox
tb = get_tb()

# read pop in from pickle file
f = open(POP_FILE, "rb")
pop = pickle.load(f)
f.close()

# read in all fitnesses for population
with open(FIT_FILE, "r") as f:
	fits = f.readlines()
	for fit, ind in zip(fits, pop):
		fit = float(fit)
		ind.fitness.values = fit,

# use to find averages of all CV values
fits = []
bound_CV = []
intersect_CV = []
axis_CV = []
gear_CV = []
# determine CV for each individual 
for ind in pop:
	mech = mechanism_from_GA(ind)
	vec = get_mechanism_vector(mech)	
	# find all different CV on mechanism
	CV_bound = check_bounding_box(mech, X_BOUND, Y_BOUND)
	CV_intersect = check_intersect_amount(mech)
	CV_axis = check_conflicting_gear_axis(mech, HOLE_SIZE)
	CV_gear = check_useless_gears(mech)	
	fits.append((ind.fitness.values[0], CV_bound, CV_intersect, CV_axis))
		
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

# DETERMINE NEXT MECHANISM TO TEST
# find best valid individual to print next
best_ind = max(valid_pop, key=lambda x: x.fitness.values[0])
best_mech = mechanism_from_GA(best_ind)
best_vec = get_mechanism_vector(best_mech)
# append vector into the file with all archive vectors
with open(ARCH_FILE, "a") as f:
	writer = csv.writer(f)
	writer.writerow(list(best_vec))
# write info for next mech into file for printing/testing
counter = 0
while(os.path.isfile(MECH_FILE + str(counter) + ".txt")):
	counter += 1
# write info for every gear to the file
with open((MECH_FILE + str(counter) + ".txt"), "w") as f:
	for g in best_mech:
		f.write(str(g))
		f.write("\n")

# perform selection on the population to maximize fitness
offspring = toolbox.select(pop, k=len(pop))
# clone offspring
offspring = toolbox.map(toolbox.clone, offspring)

# apply mutation and perform crossover
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

# generate vector for each individual in resulting pop
# write all vectors to vec file
with open(VEC_FILE, "w") as f:
	vec_list = []
	for ind in pop:
		mech = mechanism_from_GA(ind)
		vec = get_mechanism_vector(mech)
		vec_list.append(list(vec))
	# write vector contents into csv file
	writer = csv.writer(f)
	writer.writerows(vec_list)

# write population into a file with pickle
f = open(POP_FILE, "wb")
pop = pickle.dump(pop, f)
f.close()
