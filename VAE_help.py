"""contains all helper functions for the VAE implementation"""

from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.utils.data

from VAE_config import INPUT_SIZE, BATCH_SIZE

def loss_function(recon_x, x, mu, logvar):
	"""defines loss of the VAE using combined terms of reconstruction
	and KL divergence"""

	# get the reconstruction loss
	recon = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_SIZE), reduction='sum')
	# get KL loss - tells you how close latent vector is to unit gaussian
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return recon + KLD 

def train(model, optimizer, epoch):
	"""run training on the VAE

	model: the model to train
	epoch: number of epochs to run training
	"""
	
	train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True,
			download=True,transform=transforms.ToTensor()),batch_size=BATCH_SIZE,
			shuffle=True)

	model.train()
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		optimizer.zero_grad()
		recon_batch, mu, logvar = model(data)
		loss = loss_function(recon_batch, data, mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item() / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))	

def test(model, optimizer, epoch):
	"""runs the trained VAE on the testing set for mnist"""

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
		batch_size=BATCH_SIZE, shuffle=True)
	model.eval()

	with torch.no_grad():
		for i, (data, _) in enumerate(test_loader):
			recon_batch, mu, logvar = model(data)
			test_loss += loss_function(recon_batch, data, mu, logvar).item()
			if i == 0:
				n = min(data.size(0), 8)
				comparison = torch.cat([data[:n],
					recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
				#plt.imshow(comparison.numpy())
				"""
				save_image(comparison.cpu(),
					'results/reconstruction_' + str(epoch) + '.png', nrow=n)
				"""

	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
