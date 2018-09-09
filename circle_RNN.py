"""Implementation of a simple recurrent neural network create changing circular structures
that can be integrated with a CPPN to create 3D printable structures"""

import torch
import torch.nn as nn

class RNN(nn.Module):
	"""Implementation of the recurrent neural network"""

	def __init__(self, input_size, hidden_size, output_size):
		"""constructer for RNN class"""
		super(RNN, self).__init__()
		
		# store size of hidden layer
		self.hidden_size = hidden_size

		# define linear connection units in the network
		self.in2hid = nn.Linear(input_size + hidden_size, hidden_size)
		self.in2out = nn.Linear(input_size + hidden_size, output_size)
		self.outact = nn.Sigmoid()
	
	def forward(self, inputs, hidden):
		"""forward propogation function for RNN - parameters
		include both inputs and hidden unit so that RNN can be
		activated during all time steps
		"""
		
		# the current hidden state and inputs combined to make computation easily
		# hidden info is initialized to 0
		combined_in = torch.cat((inputs, hidden), 1)
		
		# yield values for hidden layer and the output layer
		hidden = self.in2hid(combined_in)
		output = self.outact(self.in2out(combined_in))
	
		# return information for hidden and output layer
		return (output, hidden)
	
