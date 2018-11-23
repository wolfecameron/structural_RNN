"""This file contains all configuration used for the evolution of the RNN
to create circular/spiral structures
"""

# NOTE: evaluation must be imported as rnn_evaluation

import numpy as np
from deap import base, tools, algorithms, creator
from scoop import futures

from deap_RNN_xover import insertion_xover, exchange_xover
from deap_RNN_evals import phase_one_eval as eval_single_obj
from deap_RNN_evals import phase_one_eval  as eval_double_obj
from deap_RNN_sel import select_binary_CV

"""The below contains all of the deap configuration used for CPPN so that it can be
called and edited from a central location"""

# constants used for deap configuration
N_IN=4
N_HID=5
N_OUT=4
MAX_POINTS = 250 # maximum num of discrete points in output structure
weights=(1.0, -1.0)
MUTPB = .15
CXPB = .5
INIT_WINDOW=2.0
POP_SIZE=50
N_GEN=500
ACT_EXP = 1.0
MAX_Y = 1.0
MAX_X = MAX_Y/2.0

# below are constants related to RNN output
OUTPUT_MIN = -1
OUTPUT_MAX = 1

# the below constants are used for gear generation
MAX_GEARS = 6
MIN_GEARS = 2
STOP_THRESHOLD = .9
PLACEMENT_THRESH = .75
RADIUS_SCALE = 10.0

# define bounding box constraints for RNN
X_BOUND = 50.0
Y_BOUND = 50.0

# defines dictionary of colors that are used for different z dimensions in mechanism visualization
C_DICT = {-9: "#CD5C5C", -8: "#C0C0C0", -7: "#000000", -6: "#800000", -5: "#008000", \
			-4: "#FFFF00", -3: "#808000", -2: "#00FF00", -1: "#FF0000", 0: "#00FFFF", \
			1: "#800080", 2: "#000080", 3: "#DC7633", 4: "#283747", 5: "#9933FF",
			6: "#1D8348", 7: "#C39BD3", 8: "#D5F5E3", 9: "#0000FF" } 

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
toolbox.register("evaluate", eval_double_obj)
toolbox.register("evaluate_single_objective", eval_single_obj)
#toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("ins_mate", insertion_xover)
toolbox.register("ex_mate", exchange_xover)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", select_binary_CV)
#toolbox.register("select", tools.selNSGA2, k=POP_SIZE)
toolbox.register("map", map)


def get_tb():
        """method to get the toolbox object that is needed for
        evolution - the toolbox is configured in this file so that
        the configuration is central"""

        return toolbox
