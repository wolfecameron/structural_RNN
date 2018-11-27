"""This file contains all custom selection methods that were implemented
for deap in the structural RNN evolution"""

from copy import deepcopy
from random import shuffle
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict

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

"""
def NSGAII_CV_tourn(tourn):
	selects one individual from the given tournament of individuals
	using NSGAII on the two fitnesses (novelty and number of nodes) of
	individuals

	the tourn parameter contains a tournament of individuals and the best
	individual is selected from this list
	
	
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
"""

def selNSGA2_cv(individuals, k):
	"""performs selection using NSGA-II on entire population with a
	constraint violation fitness value - any item that violates constaint
	is automatically dominated by one that does not
	
	majority of this code was taken from DEAP github repo and customized"""

	pareto_fronts = _sortNondominated(individuals, k)

	# determine crowding distance within each front
	for front in pareto_fronts:
		_assignCrowdingDist(front)

	chosen = list(chain(*pareto_fronts[:-1]))
	k = k - len(chosen)
	if k > 0:
		sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
		chosen.extend(sorted_front[:k])

	return chosen

def _sortNondominated(individuals, k, first_front_only=False):
	"""sort first k individuals into nondomination levels

	taken from deap and customized"""

	if k == 0:
		return []

	map_fit_ind = defaultdict(list)
	for ind in individuals:
		map_fit_ind[ind.fitness].append(ind)
	fits = list(map_fit_ind.keys())

	current_front = []
	next_front = []
	dominating_fits = defaultdict(int)
	dominated_fits = defaultdict(list)

	# Rank first Pareto front
	for i, fit_i in enumerate(fits):
		for fit_j in fits[i+1:]:
			if _dominates(fit_i, fit_j):
				dominating_fits[fit_j] += 1
				dominated_fits[fit_i].append(fit_j)
			elif _dominates(fit_j, fit_i):
				dominating_fits[fit_i] += 1
				dominated_fits[fit_j].append(fit_i)
		if dominating_fits[fit_i] == 0:
			current_front.append(fit_i)

	fronts = [[]]
	for fit in current_front:
		fronts[-1].extend(map_fit_ind[fit])
	pareto_sorted = len(fronts[-1])

	# Rank the next front until all individuals are sorted or
	# the given number of individual are sorted.
	if not first_front_only:
		N = min(len(individuals), k)
		while pareto_sorted < N:
			fronts.append([])
			for fit_p in current_front:
				for fit_d in dominated_fits[fit_p]:
					dominating_fits[fit_d] -= 1
					if dominating_fits[fit_d] == 0:
						next_front.append(fit_d)
						pareto_sorted += len(map_fit_ind[fit_d])
						fronts[-1].extend(map_fit_ind[fit_d])
			current_front = next_front
			next_front = []

	return fronts

def _assignCrowdingDist(individuals):
	"""Assign a crowding distance to each individual's fitness. The
	crowding distance can be retrieve via the :attr:`crowding_dist`
	attribute of each individual's fitness.
	
	taken from deap github repo and customized"""

	if len(individuals) == 0:
		return

	distances = [0.0] * len(individuals)
	crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

	nobj = len(individuals[0].fitness.values) - 1
	for i in range(nobj):
		crowd.sort(key=lambda element: element[0][i])
		distances[crowd[0][1]] = float("inf")
		distances[crowd[-1][1]] = float("inf")
		if crowd[-1][0][i] == crowd[0][0][i]:
			continue
		norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
		for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
			distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

	for i, dist in enumerate(distances):
		individuals[i].fitness.crowding_dist = dist


def _dominates(ind1, ind2):
	"""outputs true if ind1 dominates ind2, false o/w"""

	# get fitness vectors for each individual
	one_fit = ind1.values
	two_fit = ind2.values

	# feasible solutions dominate infeasible solutions
	if(one_fit[2] <= 0 and two_fit[2] > 0):
		return True
	
	# if both infeasible dominates if CV is less than other
	# if CV less than other, other must be nonzero!
	elif(one_fit[2] < two_fit[2]):	
		return True
	
	# if both feasible check for traditional domination
	elif(one_fit[2] <= 0 and two_fit[2] <= 0):
		# minimize hidden nodes and maximize novelty
		if(one_fit[0] > two_fit[0] and one_fit[1] < two_fit[1]):
			return True
	
	# only reaches this point if doesn't dominate
	return False

