"""This file contains all custom selection methods that were implemented
for deap in the structural RNN evolution"""

from copy import deepcopy
from random import shuffle

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

		# always select inds who satisfy constraints if only one satisfies
		# select for highest fitness/lowest constraint violation
		if(ind1_fit[1] <= 0 and ind2_fit[1] > 0):
			new_pop.append(ind1)
		elif(ind1_fit[1] > 0 and ind2_fit[1] <= 0):
			new_pop.append(ind2)
		elif(ind1_fit[1] <= 0 and ind2_fit[1] <= 0):
			if(ind1_fit[0] > ind2_fit[0]):
				new_pop.append(ind1)
			else:
				new_pop.append(ind2)
		else:
			if(ind1_fit[1] < ind2_fit[1]):
				new_pop.append(ind1)
			else:
				new_pop.append(ind2)

	return new_pop
