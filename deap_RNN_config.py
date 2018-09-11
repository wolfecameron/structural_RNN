"""This file contains all configuration used for the evolution of the RNN
to create circular/spiral structures
"""

# NOTE: evaluation must be imported as rnn_evaluation

import numpy as np
from deap import base, tools, algorithms, creator
from scoop import futures




"""The below contains all of the deap configuration used for CPPN so that it can be
called and edited from a central location"""

# constants used for deap configuration
N_IN=4 
N_HID=15
N_OUT=2
weights=(1.0,)
INIT_WINDOW=.5
POP_SIZE=100

# total number of weights present in RNN
TOTAL_WEIGHTS=(N_IN + N_HID)*N_HID + (N_HID*N_OUT)

#create types needed for deap
creator.create("FitnessMax", base.Fitness, weights=weights)
creator.create("Individual", list, fitness=creator.FitnessMax)

# initialize the toolbox
toolbox = base.Toolbox()

# register function to create individual in the toolbox
toolbox.register("get_init_weight", np.random.uniform, -INIT_WINDOW,INIT_WINDOW)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
		toolbox.get_init_weight, n=TOTAL_WEIGHTS)


# register function to create population in the toolbox
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

# register all functions needed for evolution in the toolbox
#toolbox.register("evaluate", rnn_evaluation)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("map", futures.map)


def get_tb():
        """method to get the toolbox object that is needed for
        evolution - the toolbox is configured in this file so that
        the configuration is central"""

        return toolbox
