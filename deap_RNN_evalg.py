"""contains all methods for evolution of RNNs"""

import numpy as np

def apply_mutation(pop, tb, mutpb):
	"""applies mutations to the population
	
	:param tb: deap toolbox that contains mutate method
	"""

	for ind in pop:
		if np.random.uniform() <= mutpb:
			tb.mutate(ind)
			del ind.fitness.values	

def apply_crossover(pop, tb, cxpb, num_in, num_out):
	"""applies crossover to population"""

	for child1, child2 in zip(pop[::2], pop[1::2]):
		# find number of hidden nodes in each individual
		one_n = child1.h_nodes
		two_n = child2.h_nodes
			
		# find cutoff for hidden/output weights
		one_hid = (one_n + num_in)*one_n + one_n # (num_hid + n_in)*n_hid + n_hid 
		one_end = one_hid + one_n*num_out + num_out # hidden weights + n_hid*n_out + n_out
		two_hid = (two_n + num_in)*two_n + two_n
		two_end = two_hid + two_n*num_out + num_out
		rand = np.random.uniform()
			
		# 50-50 chance of using either crossover operator
		if rand <= (cxpb/2.0):
			child1[ :one_hid], child2[ :two_hid] = tb.ins_mate(child1[ :one_hid], child2[ :two_hid])
			child1[one_hid: one_end], child2[two_hid: two_end] = tb.ins_mate(child1[one_hid: one_end], child2[two_hid: two_end])
			del child1.fitness.values
			del child2.fitness.values
		elif (cxpb/2.0) < rand <= cxpb:
			child1[ :one_hid], child2[ :two_hid] = tb.ex_mate(child1[ :one_hid], child2[ :two_hid], cxpb)
			child1[one_hid: one_end], child2[two_hid: two_end] = tb.ex_mate(child1[one_hid: one_end], child2[two_hid: two_end], cxpb)
			del child1.fitness.values
			del child2.fitness.values
