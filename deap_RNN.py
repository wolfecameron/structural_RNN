"""Main deap evolution file for evolving the RNN to create different types of circular
structures - uses deap to evolve sets of weights that are inputted into RNN to yield output
"""

from deap_RNN_config import get_tb

# import toolbox from config file
toolbox = get_tb()
