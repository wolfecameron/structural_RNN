"""implements the second phase of evolution utilizing the surrogate to assign fitnesses to individuals
within the population

entire file only runs one generation of the evolution so that evolution can be controlled by the
matlab surrogate code
"""

import pickle

from deap_RNN_config import get_tb, MUTPB, N_IN, N_OUT, CXPB
from deap_RNN_config import FIT_FILE, POP_FILE
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

# TODO: assess CV to change fitness
# select population based on fitness from surrogate
pop = tb.select(pop)

# apply mutation and perform crossover
apply_mutation(pop, tb, MUTPB)
apply_crossover(pop, tb, CXPB, N_IN, N_OUT)
	

# generate vector for each individual in resulting pop
# find the best individual, append that vector into the archive vector file
