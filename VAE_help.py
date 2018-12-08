"""contains all helper functions for the VAE implementation"""

from torch.nn import functional as F

from VAE_config import INPUT_SIZE

def loss_function(recon_x, x, mu, logvar):
	"""defines loss of the VAE using combined terms of reconstruction
	and KL divergence"""

	# get the reconstruction loss
	recon = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_SIZE), reduction='sum')
	# get KL loss - tells you how close latent vector is to unit gaussian
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return recon + KLD 

	
