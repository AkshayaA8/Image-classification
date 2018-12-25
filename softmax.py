import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        
        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''
        super(Softmax, self).__init__()
        
        self.im_size = im_size
        self.n_classes = n_classes
        self.softmax = nn.Linear(in_features=np.prod(im_size), out_features=n_classes)
        

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
        scores = self.softmax(output)
       
        return scores

