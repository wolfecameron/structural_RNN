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
		self.hid2out = nn.Linear(hidden_size, output_size)
		
		# RNN hidden layer typically activated with Tanh activation function
		self.hid_act = nn.Tanh()
		
		# activate output with ReLU so that outputs always positive
		self.out_act = nn.Sigmoid()
	
	def forward(self, inputs, hidden):
		"""forward propogation function for RNN - parameters
		include both inputs and hidden unit so that RNN can be
		activated during all time steps
		"""
		
		# the current hidden state and inputs combined to make computation easily
		# hidden info is initialized to 0
		combined_in = torch.cat((inputs, hidden), 1)
		
		# yield values for hidden layer and the output layer
		hidden = self.hid_act(self.in2hid(combined_in))
		output = self.out_act(self.hid2out(hidden))
		
		# return information for hidden and output layer
		return (output, hidden)


if __name__ == '__main__':
	"""Used for simple testing"""
	
	# instantiate the RNN
	rnn = RNN(4, 4, 2)
	print(rnn.in2hid.weight.data.numpy().shape)	
	print(rnn.hid2out.weight.data.numpy().shape)
	'''	
	# instantiate all needed variables for running RNN
	r = 50.0
	theta = 0.0
	hidden = torch.zeros(1, rnn.hidden_size)
	all_pos = []
	dr = 0.0
	dt = 0.0
	while(r > 0):
		print("Current R: {0}".format(str(r)))
		rnn_pos = (r, theta)
		all_pos.append(rnn_pos)
		rnn_input = [[r, theta, dr, dt]]
		outs, hidden = rnn.forward(torch.Tensor(rnn_input), hidden)
		dr, dt = outs.data[0][0], outs.data[0][1]
		print(dr)
		print(dt)
		input()
		r -= dr
		if(r <= 0):
			r = 0
		theta += dt
		theta %= 2.0
	all_pos.append([[r, theta]])
	print(all_pos)
	'''
