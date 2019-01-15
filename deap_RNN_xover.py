"""contains implementation of custom crossover operaters for genetic algorithm"""

import numpy as np
from copy import deepcopy

# set seed number in numpy for reproducing results
seed_f = open("seed.txt", "r")
np.random.seed(int(seed_f.readlines()[0]))
seed_f.close()

def insertion_xover(ind1, ind2):
	"""performs insertion crossover between two sets of weights, where
	a given portion of weights is exchanged between the two lists
	somewhere at beginning, middle, or end of the lists

	list can be different sizes
	"""

	# find the max length section to crossover
	max_size = min(len(ind1), len(ind2))
	insertion_size = np.random.randint(1, max_size + 1)

	# pick the starting index for crossover on both lists
	first_ind = np.random.randint(0, len(ind1) - insertion_size + 1)
	second_ind = np.random.randint(0, len(ind2) - insertion_size + 1)

	# swap the sections between to individuals
	tmp = ind1[first_ind: first_ind + insertion_size]
	ind1[first_ind: first_ind + insertion_size] = ind2[second_ind: second_ind + insertion_size]
	ind2[second_ind: second_ind + insertion_size] = tmp
	
	return ind1, ind2

def exchange_xover(ind1, ind2, cxpb):
	"""performs an exchange crossover between the two individuals, which
	does a uniform crossover of each weight between individuals

	the individuals can be of different length

	Parameters:
	cxpb -- probability that each weight will be switched in uniform crossover
	"""
	
	# find length of the smaller set of weights
	min_size = min(len(ind1), len(ind2))
	
	# pick starting index in each list
	first_ind = np.random.randint(0, len(ind1) - min_size + 1)
	second_ind = np.random.randint(0, len(ind2) - min_size + 1)
	
	while(first_ind < len(ind1) and second_ind < len(ind2)):	
		# swap weights if random number below desired probability
		if(np.random.uniform() <= cxpb):
				tmp = ind1[first_ind]
				ind1[first_ind] = ind2[second_ind]
				ind2[second_ind] = tmp
		
		first_ind += 1
		second_ind += 1
	
	return ind1, ind2
	

if __name__ == '__main__':
	"""used for quick testing"""

	ind1 = [1, 4, 6, 8]
	ind2 = [9, 12, 13, 16]
	ind1_og = deepcopy(ind1)
	ind2_og = deepcopy(ind2)
	ind1_n, ind2_n = exchange_xover(ind1, ind2, .25)
	print("{0} ---> {1}".format(str(ind1_og), str(ind1_n)))
	print("{0} ---> {1}".format(str(ind2_og), str(ind2_n)))
