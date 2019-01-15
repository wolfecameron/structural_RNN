"""This file contains all configuration used for the evolution of the RNN
to create circular/spiral structures
"""

# NOTE: evaluation must be imported as rnn_evaluation

import numpy as np
from deap import base, tools, algorithms, creator
from scoop import futures
import random

from deap_RNN_xover import insertion_xover, exchange_xover
from deap_RNN_evals import phase_one_eval as eval_single_obj
from deap_RNN_evals import phase_one_eval  as eval_double_obj
from deap_RNN_sel import selNSGA2_cv

"""The below contains all of the deap configuration used for CPPN so that it can be
called and edited from a central location"""

# set seed number in numpy for reproducing results
seed_f = open("seed.txt", "r")
seed_val = int(seed_f.readlines()[0])
np.random.seed(seed_val)
random.seed(seed_val)
seed_f.close()

# constants used for deap configuration
NUM_UNIQUE_GEARS = 6
N_IN=NUM_UNIQUE_GEARS + 2
N_HID=8
N_OUT=N_IN
weights=(1.0, )
MUTPB = .3
CXPB = .15
INIT_WINDOW=1.0
POP_SIZE=50
N_GEN=20
ACT_EXP = 1.0
MAX_Y = 1.0
MAX_X = MAX_Y/2.0
TOURN_SIZE = 3
GEAR_RADII = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0]

# 20 24, 8 20, 
# constants for creating slot inserts for gears
GEAR_DISTS = {(8.0, 8.0): 13.0, (8.0, 12.0): 17.0, (8.0, 16.0): 20.0, (8.0, 20.0): 25.0, (8.0,24.0): 29.0, (8.0, 28.0): 33.0\
				,(12.0, 12.0): 20.0, (12.0, 16.0): 25.0, (12.0, 20.0): 29.0, (12.0, 24.0): 33.0, (12.0, 28.0): 37.0\
				,(16.0, 16.0): 29.0, (16.0, 20.0): 33.0, (16.0, 24.0): 37.0, (16.0, 28.0): 42.0, (20.0, 20.0): 37.0\
				,(20.0, 24.0): 42.0, (20.0, 28.0): 45.0, (24.0, 24.0): 45.0, (24.0, 28.0): 49.0, (28.0, 28.0): 53.0}
SLOT_LEN = 90
SLOT_HT = 3.5
SLOT_T = 1.0
INIT_OFFSET = 8
DIST_FROM_CENT = 1.5
HOLE_R = 1.75
SLOT_HOLE_LEN = 1.5
SLOT_HOLE_HT = 1.5

# below are constants related to RNN output
OUTPUT_MIN = -1
OUTPUT_MAX = 1

# the below constants are used for gear generation
MAX_GEARS = 6
MIN_GEARS = 2
STOP_THRESHOLD = .5
PLACEMENT_THRESH = .5
RADIUS_SCALE = 15.0
CIRCULAR_PITCH = 6.0
PRESSURE_ANGLE = 28.0
GEAR_THICKNESS = 6.0
HOLE_SIZE = 5.0
MIN_TEETH = 20 # minimum number of teeth a hollow gear can have

# constants for pickle
POP_FILE = "POP_FILES/pop.txt"
VEC_FILE = "POP_FILES/vec.csv"
FIT_FILE = "POP_FILES/fit.csv"
ARCH_FILE = "POP_FILES/arch_vec.csv"
MECH_FILE = "POP_FILES/MECHANISMS/mech"

# constants configuring the size of 3D printer
BED_WIDTH = 400
PADDING_RATIO = 1.35

# define bounding box constraints for RNN
X_BOUND = 75.0
Y_BOUND = 400.0

# defines dictionary of colors that are used for different z dimensions in mechanism visualization
C_DICT = {-9: "#CD5C5C", -8: "#C0C0C0", -7: "#000000", -6: "#800000", -5: "#008000", \
			-4: "#FFFF00", -3: "#808000", -2: "#00FF00", -1: "#FF0000", 0: "#00FFFF", \
			1: "#800080", 2: "#000080", 3: "#DC7633", 4: "#283747", 5: "#9933FF",
			6: "#1D8348", 7: "#C39BD3", 8: "#D5F5E3", 9: "#0000FF" } 

# total number of weights present in RNN
TOTAL_WEIGHTS=(N_IN + N_HID)*N_HID + (N_HID*N_OUT) + N_HID + N_OUT

# create types needed for deap
creator.create("FitnessMulti", base.Fitness, weights=weights)
creator.create("Individual", list, fitness=creator.FitnessMulti, CV=0.0, h_nodes=-1)

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
toolbox.register("ins_mate", insertion_xover)
toolbox.register("ex_mate", exchange_xover)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, k=POP_SIZE, tournsize=3)
#toolbox.register("select", tools.selNSGA2, k=POP_SIZE)
toolbox.register("map", map)


def get_tb():
        """method to get the toolbox object that is needed for
        evolution - the toolbox is configured in this file so that
        the configuration is central"""

        return toolbox
