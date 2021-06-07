import sys
import argparse

import torch
from torch import nn, optim
from data import mnist
from model import MNIST_NET
import matplotlib.pyplot as plt


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.003)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MNIST_NET().to(device)
        trainloader, _ = mnist()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        steps = 0
        running_loss = 0
        train_loss = []
        for e in range(50):
            # Model in training mode, dropout is on
            print('Epoch: ', e)
            model.train()
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)
                steps += 1
                
                optimizer.zero_grad()
                
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                train_loss.append(loss.item())
            
        torch.save(model.state_dict(), 'checkpoint.pth')
        plt.figure(figsize = (8,12))
        plt.plot(range(1,steps+1), train_loss)
        plt.xlabel('Step')
        plt.ylabel('Training loss')
        plt.title('Training loss - MNIST model')
        plt.savefig('train_loss.png')

    def evaluate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = torch.load(args.load_model_from)
        else:
            model = MNIST_NET().to(device)
            state_dict = torch.load('checkpoint.pth')
            model.load_state_dict(state_dict)

        _, testloader = mnist()

        criterion = nn.CrossEntropyLoss()
        accuracy = []
        test_loss = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ## Calculating the accuracy 
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == output.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy.append(equality.type_as(torch.FloatTensor()).mean())

        print('Test accuracy', torch.mean(torch.stack(accuracy)))

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    