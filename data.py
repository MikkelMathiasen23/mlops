import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

def mnist():
  device = torch.device('cuda') 	
  transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
  # Download and load the training data
  trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

  validationset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
  valloader = torch.utils.data.DataLoader(validationset, batch_size=64, shuffle=True)

  return trainloader, valloader
