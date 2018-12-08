import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.distributions as tdist

from VAE_config import SEED, BATCH_SIZE, LOG_INTERVAL, EPOCHS, LATENT_DIMS
from VAE_config import RELU_SIZE, INPUT_SIZE
from VAE_help import train, test
from gear_VAE import VAE
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# set seed number for pytorch
torch.manual_seed(SEED)

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=.001)

# run VAE
for epoch in range(1, EPOCHS):
		train(model, optimizer, epoch)
		test(model, optimizer, epoch)
		with torch.no_grad():
			sample = torch.randn(64, 20)
			sample = model.decode(sample)
			plt.imshow(sample.view(64, 1, 28, 28).numpy())
