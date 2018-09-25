"""This file contains all configuration used for the evolution of the RNN
to create circular/spiral structures
"""

# NOTE: evaluation must be imported as rnn_evaluation

import numpy as np
from deap import base, tools, algorithms, creator
from scoop import futures

from deap_RNN_evals import  loops_and_novelty_eval as rnn_evaluation


"""The below contains all of the deap configuration used for CPPN so that it can be
called and edited from a central location"""

# constants used for deap configuration
N_IN=2 
N_HID=10
N_OUT=2
#RADIUS = 20.0
MAX_POINTS = 250 # maximum num of discrete points in output structure
weights=(1.0, 1.0)
MUTPB = .15
CXPB = .05
INIT_WINDOW=.1
POP_SIZE=50
N_GEN=50
#MIN_THICKNESS = .5
#MAX_THICKNESS = 5.5
ACT_EXP = .1

# total number of weights present in RNN
TOTAL_WEIGHTS=(N_IN + N_HID)*N_HID + (N_HID*N_OUT) + N_HID + N_OUT

#create types needed for deap
creator.create("FitnessMulti", base.Fitness, weights=weights)
creator.create("Individual", list, fitness=creator.FitnessMulti)

# initialize the toolbox
toolbox = base.Toolbox()

# register function to create individual in the toolbox
toolbox.register("get_init_weight", np.random.uniform, -INIT_WINDOW,INIT_WINDOW)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
		toolbox.get_init_weight, n=TOTAL_WEIGHTS)


# register function to create population in the toolbox
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

# register all functions needed for evolution in the toolbox
toolbox.register("evaluate", rnn_evaluation)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
#toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selNSGA2, k=POP_SIZE)
toolbox.register("map", map)


def get_tb():
        """method to get the toolbox object that is needed for
        evolution - the toolbox is configured in this file so that
        the configuration is central"""

        return toolbox
