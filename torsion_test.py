"""Implements test for circular RNN - tries to create a circular structure with
as many loops as possible in the final output structure"""


import torch
from torch import optim
import torch.nn as nn
from circle_RNN import RNN

def main(num_in, num_hid, num_out, radius):
	"""Main method that runs the test"""

	# instantiate the RNN
	rnn = RNN(num_in, num_hid, num_out)
	
	
	# train the rnn
	training_iterations = 20000
	target_dr = 1.0
	target_dt = .5
	rnn = train_rnn(rnn, radius, training_iterations, target_dr, target_dt)
	
	
	# get output of the RNN
	max_t = 200
	positions = run_rnn(rnn, radius, max_t)
	
	print(positions)   


def run_rnn(rnn, radius, max_t, verbose=False):
	"""runs the rnn to create a new loop
	structure and returns polar coordinates
	of every point in the loop"""
	
	# initialize all tracking values that are needed
	# to create a structure with the rnn
	r = radius
	theta = 0.0
	hidden = torch.zeros(1, rnn.hidden_size)
	all_positions = []
	dr = 0.0
	dt = 0.0
	curr_t = 0 # track current t so that it does not exceed max
	
	# run rnn until candidate structure reaches the origin
	while (r > 0 and curr_t < max_t):
		# add current position into structure history
		rnn_pos = (r, theta)
		all_positions.append(rnn_pos)
	
		# get input and activate rnn at current timestep
		rnn_input = [[r, theta, dr, dt]]
		outs, hidden = rnn.forward(torch.Tensor(rnn_input), hidden)
		dr, dt = outs.data[0][0].item(), outs.data[0][1].item()
		
		# print information
		if verbose:
			print("Current R: {0}".format(str(r)))
			print("dR: {0}".format(str(dr)))
			print("dT: {0}".format(str(dt)))
		
		# update the current position of the structure
		r -= dr
		theta += dt
		theta = theta % 2.0
		
		# increment the current time step
		curr_t += 1
	
	# append the last position into the list
	all_positions.append((r, theta))		
	
	return all_positions

def train_rnn(rnn, radius, num_it, dr_target, dt_target):
	"""Trains RNN to output a certain value continually for dR
	and dT to create a circlar-looking structure
	
	Parameters:
	rnn -- the rnn that is being trained
	radius -- starting radius of the polar coordinates
	num_it -- total number of training iterations
	dr_target -- desired value of dr on each time step
	dt_target -- desired value of dt on each time step
	"""
	
	# define criterion for training/backprop
	criterion = nn.MSELoss()
	optimizer = optim.SGD(rnn.parameters(), lr=0.01)
	target_tensor = torch.Tensor([[dr_target, dt_target]])

	# create a counter for number of iterations
	total_it = 0

	# continue running until desired number of iterations is met
	while(total_it < num_it):
		# initialize all variable needed to run rnn
		r = radius
		theta = 0.0
		hidden = torch.zeros(1, rnn.hidden_size)
		dr = 0.0
		dt = 0.0
		curr_it = 0
		
		# zero the gradients
		rnn.zero_grad()
		outs = None
		
		# go until radius reaches 0	
		while(r > 0 and curr_it < 200):	
	
			# get output of rnn until radius reaches 0
			rnn_input = [[r, theta, dr, dt]]
			outs, hidden = rnn.forward(torch.Tensor(rnn_input), hidden)
	
			# find dr and dt values
			dr = outs[0][0].item()
			dt = outs[0][1].item()

			# update position
			r -= dr
			if (r < 0):
				r = 0
			theta += dt
			theta = theta % 2
			
			# increment iterations
			curr_it += 1	
		
		# run update on rnn parameters
		print(outs)
		loss = criterion(outs, target_tensor)
		loss.backward(retain_graph=True)
		optimizer.step()				

		# increment total iterations
		total_it += curr_it

	return rnn
 			
if __name__ == '__main__':
	"""main loop to run code"""

	main(4, 4, 2, 50)
