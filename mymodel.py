import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        
        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        
        self.im_size = im_size
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size


        self.conv1 = nn.Conv2d(in_channels=im_size[0], out_channels=hidden_dim, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size)
        self.fc = nn.Linear(in_features = int(self.hidden_dim*((self.im_size[1]-(2*self.kernel_size)+2)/2)*((self.im_size[2]-(2*self.kernel_size)+2)/2)), out_features=n_classes)
        

    def forward(self, images):
        '''
        
        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        
        x = self.conv1(images)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, int(self.hidden_dim*((self.im_size[1]-(2*self.kernel_size)+2)/2)*((self.im_size[2]-(2*self.kernel_size)+2)/2)))
        scores = self.fc(x)
        
        return scores

