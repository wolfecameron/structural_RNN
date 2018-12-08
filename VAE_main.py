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
from gear_VAE import VAE

# set seed number for pytorch
torch.manual_seed(SEED)

model = VAE()
opt = optim.Adam(model.parameters(), lr=.001)
