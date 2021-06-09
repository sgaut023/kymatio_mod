"""Contains all the pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier

Exceptions:
    InvalidInitializationException --
    InvalidArchitectureError --

Functions: 

Classes: 
    sn_Identity -- computes the identity function in forward pass
    sn_HybridModel -- combinations of a scattering and other nn.modules
    sn_ScatteringBase -- a scattering network
"""

import torch.nn as nn

from numpy.core.numeric import False_


class sn_HybridModel(nn.Module):
    """An nn.Module incorporating scattering an a learnable network"""

    def __str__(self):
        return str(self.scatteringBase)

    def __init__(self,scatteringBase,top,use_cuda=True):
        super(sn_HybridModel,self).__init__()
        if use_cuda:
            self.scatteringBase = scatteringBase.cuda()
            self.top = top.cuda()
        else:
            self.scatteringBase = scatteringBase.cpu()
            self.top = top.cpu()

    def parameters(self):
        """implements parameters method allowing the use of scatteringBase's method """
        temp = [{'params': list(self.top.parameters())}]
        temp.extend(list(self.scatteringBase.parameters()))
        return temp

    def forward(self,inp):
        return self.top(self.scatteringBase(inp))

    def train(self):
        self.scatteringBase.train()
        self.top.train()

    def eval(self):
        self.scatteringBase.eval()
        self.top.eval()

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        return self.scatteringBase.countLearnableParams() \
            + self.top.countLearnableParams()

    def showParams(self):
        """prints shape of all parameters and is_leaf"""
        for x in self.parameters():
            if type(x['params']) == list:
                for tens in x['params']:
                    print(tens.shape,tens.is_leaf)
            else:
                print(x['params'].shape,x['params'].is_leaf)


