"""This file contains all custom selection methods that were implemented
for deap in the structural RNN evolution"""

from copy import deepcopy
from random import shuffle

from deap_RNN_help import dominates, get_crowding_distance

def select_binary_CV(pop):
	"""this method performs binary selection on a population
	with contraint violation - individuals that satisfy constraint
	are always selected over those that do not. If both do not
	satisfy then the one that violates less is selected"""

	# create two version of pop and randomly shuffle them
	pop_o = deepcopy(pop)
	shuffle(pop_o)
	shuffle(pop)

	# collect selected population in new list
	new_pop = []	

	# go through each pair of individuals and select
	for ind1, ind2 in zip(pop, pop_o):
		ind1_fit = ind1.fitness.values
		ind2_fit = ind2.fitness.values
		#print(ind1_fit)
		#print(ind2_fit)
	
		# always select inds who satisfy constraints if only one satisfies
		# select for highest fitness/lowest constraint violation
		if(ind1_fit[1] == 0 and ind2_fit[1] > 0):
			new_pop.append(ind1)
			#print("ind1 selected")
		elif(ind1_fit[1] > 0 and ind2_fit[1] == 0):
			new_pop.append(ind2)
			#print("ind2 selected")
		elif(ind1_fit[1] == 0 and ind2_fit[1] == 0):
			if(ind1_fit[0] > ind2_fit[0]):
				new_pop.append(ind1)
				#print("ind1 selected")
			else:
				new_pop.append(ind2)
				#print("ind2 selected")
		else:
			if(ind1_fit[1] < ind2_fit[1]):
				new_pop.append(ind1)
				#print("ind1 selected")
			else:
				new_pop.append(ind2)
				#print("ind2 selected")
		#input()

	return new_pop

def NSGAII_CV_tourn(tourn):
	"""selects one individual from the given tournament of individuals
	using NSGAII on the two fitnesses (novelty and number of nodes) of
	individuals

	the tourn parameter contains a tournament of individuals and the best
	individual is selected from this list
	"""
	
	# initialize pareto front as empty
	pareto_front = []
	
	# populate the first pareto front using individuals in tourn
	l_ind = 0
	while(l_ind < len(tourn)):
		r_ind = 0
		dominated = False
		while(not dominated and r_ind < len(tourn)):
			d = dominates(tourn[r_ind], tourn[l_ind])
			if(d):
				dominated = True
			r_ind += 1
		if(not dominated):
			pareto_front.append(tourn[l_ind])
		l_ind += 1
	
	# if only one dominating solution then return it
	if(len(pareto_front) == 1):
		return pareto_front[0]
	# if all individuals in pareto front are not viable
	# return the one with lowest constraint violation
	elif(pareto_front[0].fitness.values[2] > 0):
		return min(pareto_front, key=lambda x: x.fitness.values[2])
	
	crowd_dist_solutions = []
	# calculate the crowding distance for each solution
	l_ind = 0
	while(l_ind < len(pareto_front)):
		r_ind = 0
		crowd_dists = []
		while(r_ind < len(pareto_front)):
			if(l_ind != r_ind):
				crowd_dists.append(get_crowding_distance(pareto_front[l_ind], pareto_front[r_ind]))
			r_ind += 1
		crowd_dists = sorted(crowd_dists)
		# track individuals crowding distance so one with max crowding distance can be selected
		crowd_dist_solutions.append((pareto_front[l_ind], crowd_dists[0]))
		l_ind += 1
	
	# return individual with highest crowding distance
	return max(crowd_dist_solutions, key=lambda x: x[1])[0]
