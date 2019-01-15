"""Implementation of a simple recurrent neural network create changing circular structures
that can be integrated with a CPPN to create 3D printable structures"""

import torch
import torch.nn as nn
import numpy as np

# set seed number in numpy for reproducing results
seed_f = open("seed.txt", "r")
np.random.seed(int(seed_f.readlines()[0]))
seed_f.close()

class RNN(nn.Module):
	"""Implementation of the recurrent neural network"""

	def __init__(self, input_size, hidden_size, output_size):
		"""constructer for RNN class"""
		super(RNN, self).__init__()
		
		# store size of hidden layer
		self.hidden_size = hidden_size

		# define linear connection units in the network
		self.in2hid = nn.Linear(input_size + hidden_size, hidden_size)
		self.hid2out = nn.Linear(hidden_size, output_size)
		
		# RNN hidden layer typically activated with Tanh activation function
		self.hid_act = nn.Tanh()
		
		# activate output with ReLU so that outputs always positive
		self.out_act_tanh = nn.Tanh()
		self.out_act_sm = nn.Softmax(dim=1)
	
	def forward(self, inputs, hidden, activation_exponent):
		"""forward propogation function for RNN - parameters
		include both inputs and hidden unit so that RNN can be
		activated during all time steps
		"""
		
		# the current hidden state and inputs combined to make computation easily
		# hidden info is initialized to 0
		combined_in = torch.cat((inputs, hidden), 1)
		
		# yield values for hidden layer and the output layer
		hidden = self.hid_act(self.in2hid(combined_in))
		output = self.out_act(activation_exponent*self.hid2out(hidden))
		# return information for hidden and output layer
		return (output, hidden)

	def forward_softmax(self, inputs, hidden, num_sm, activation_exponent):
		"""different forward propagation method for RNN - a certain number of
		outputs are activated with softmax while others are activated with tanh
		
		:param inputs: the input values into forward prop
		:param hidden: values for previous hidden layer
		:param num_sm: the number of outputs values for which softmax is applied,
			tanh is applied to all others
		
		:returns: output of forward prop and new values for hidden layer
		"""

		# combine hidden with inputs
		combined_in = torch.cat((inputs, hidden), 1)
		
		# get hidden and output values - outputs not yet activated
		hidden = self.hid_act(self.in2hid(combined_in))
		output = self.hid2out(hidden)
	
		# separate output into softmax and tanh activation
		out_sm = self.out_act_sm(output[:, :num_sm])
		out_tanh = self.out_act_tanh(activation_exponent*output[:, num_sm:])
		output = torch.cat((out_sm, out_tanh), 1)

		return (output, hidden)

if __name__ == '__main__':
	"""Used for simple testing"""
	
	# instantiate the RNN
	rnn = RNN(4, 4, 2)
	print(rnn.in2hid.weight.data.numpy().shape)	
	print(rnn.hid2out.weight.data.numpy().shape)

