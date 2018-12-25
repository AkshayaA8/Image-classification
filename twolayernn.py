import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable


class TwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(TwoLayerNN, self).__init__()
       
        self.im_size = im_size
        self.softmax1 = nn.Linear(in_features=np.prod(im_size), out_features=hidden_dim)
        self.softmax2 = nn.Linear(in_features=hidden_dim, out_features=n_classes)
        

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
        
        output = images.view(-1, np.prod(self.im_size))
        output = self.softmax1(output)
        scores = self.softmax2(output)
        
        return scores

