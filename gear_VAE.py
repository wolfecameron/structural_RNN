"""this file contains an implementation of a VAE to be used to sample weights of RNN to generate
distributions of gear mechanisms"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.distributions as tdist

from deap_RNN_config import SEED, BATCH_SIZE, LOG_INTERVAL, EPOCHS, LATENT_DIMS
from deap_RNN_config import RELU_SIZE, INPUT_SIZE

# set seed number for pytorch
torch.manual_seed(SEED)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # ENCODER
		# linear unit to yield relu layer
		self.enc_relu = nn.Linear(INPUT_SIZE, RELU_SIZE)
		self.relu = nn.ReLU()
		
		# two linear units to yield the mean and STD latent vectors
		self.enc_mu = nn.Linear(RELU_SIZE, ZDIMS)  
		self.enc_sig = nn.Linear(RELU_SIZE, ZDIMS)
		# this last layer bottlenecks through ZDIMS connections

		# DECODER
		# upsample the relu layer from the latent vector
		self.dec_relu = nn.Linear(ZDIMS, RELU_SIZE)
		# upsample to original input size
		self.dec_out  = nn.Linear(RELU_SIZE, INPUT_SIZE)
		self.sigmoid = nn.Sigmoid()

	def encode(self, inputs):
		"""encode the inputted example, yielding two vectors of size Z
		representing mean and STD of the latent variables"""

		# pass through first set of weights and apply relu
		relu_layer = self.relu(self.enc_relu(inputs))

		# return the intermediate layer multiplied by weights to yield two latent vectors
		return self.enc_mu(relu_layer), self.enc_sig(relu_layer)

	def decode(self, latent_vec):
		"""upsamples a given latent vector to recreate the input or
		create new ouput"""

		return self.sigmoid(self.dec_out(self.dec_relu(latent_vec)))

	def sample(self, mu, sigma, training=False):
		"""sample from the vectors of mean and standard deviation created by
		the encoder
		
		training: if not currently training, mu vector is returned because those
		values are the most likely
		
		return: a vector of samples from each of the normal distributions
		"""

		if training:
			# sample from each normal distribution
			print(mu.element_size())
			distrib = tdist.Normal(mu, sigma)
			sample = distrib.sample(mu.element_size())			
			print(sample.element_size()) # remove comment after you verify it's the same!
			return sample

		else:
			return mu

	def encode_decode(self, x):
		"""takes and inputted vector of size [1, 784] and passes it through
		both the encoder and decoder, yielding an output"""

		# pass input through encoder, sample result, and pass latent vec through decoder
		mu, sigma = self.encode(x)
		latent_vec = self.sample(mu, sigma)
		return self.decode(latent_vec)
