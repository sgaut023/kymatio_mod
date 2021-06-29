"""Contains all the 'top' pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier

Exceptions:
    InvalidInitializationException --
    InvalidArchitectureError --

Functions: 
    create_scatteringExclusive -- creates scattering parameters
    topModelFactory -- creates different models based on input

Classes: 
    sn_CNN -- CNN fitted for scattering input
    sn_LinearLayer -- Linear layer fitted for scattering input
    sn_MLP -- multilayer perceptron fitted for scattering input
"""

from numpy.core.numeric import False_

import torch.nn as nn

from .sn_models_exceptions import InvalidInitializationException, InvalidArchitectureError


class InvalidInitializationException(Exception):
    """Error thrown when an invalid initialization scheme is passed"""
    pass

class InvalidArchitectureError(Exception):
    """Error thrown when an invalid architecture name is passed"""
    pass

def topModelFactory(base,architecture,num_classes, width=8, average=False, use_cuda=True):
    """factory for the creation of different model architectures associated to a scattering base"""

    if architecture.lower() == 'cnn':
        return sn_CNN(
            base.n_coefficients, k=width, num_classes=num_classes, standard=False
        )
    elif architecture.lower() == 'mlp':
        return sn_MLP(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
            use_cuda=use_cuda
        )
    elif architecture.lower() == 'linear_layer':
        return sn_LinearLayer(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
            average=average
        )
    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()


class sn_MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8, use_cuda=True):
        super(sn_MLP,self).__init__()
        selfnum_classes =num_classes
        if use_cuda:
            self.cuda()

        fc1=  nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients),  512)

        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.n_coefficients*3,eps=1e-5,affine=True),
            fc1,
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        '''Forward pass'''
        x = x.view(x.shape[0], -1)
        return self.layers(x)

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        count = 0
        for t in self.parameters():
            count += t.numel()
        return count



class sn_LinearLayer(nn.Module):
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8, average=False, use_cuda=True):
        super(sn_LinearLayer,self).__init__()
        self.n_coefficients = n_coefficients
        self.num_classes = num_classes
        self.average= average
        if use_cuda:
            self.cuda()

        if self.average:
            self.fc1 = nn.Linear(int(3*n_coefficients), num_classes)
        else:
            self.fc1 = nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients), num_classes)

        self.bn0 = nn.BatchNorm2d(self.n_coefficients*3,eps=1e-5,affine=True)


    def forward(self, x):
        # x = x[:,:, -self.n_coefficients:,:,:]
        x = self.bn0(x)
        if self.average:
            x = x.mean(dim=(2,3))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""

        count = 0
        for t in self.parameters():
            count += t.numel()
        return count


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class sn_CNN(nn.Module):
    def __init__(self, in_channels, k=8, n=4, num_classes=10, standard=False):
        super(sn_CNN, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_channels*3,eps=1e-5,affine=True)

        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.in_channels = in_channels
        self.num_classes =num_classes
        in_channels = in_channels * 3
        if standard:

            self.init_conv = nn.Sequential(
                nn.Conv2d(3, self.ichannels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.layer1 = self._make_layer(BasicBlock, 16 * k, n)
            self.standard = True
        else:
            self.K = in_channels
            self.init_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
                nn.Conv2d(in_channels, self.ichannels,
                      kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.standard = False

        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.standard:
            pass
            # x = x[:,:, -self.in_channels:,:,:]\
            # print(x.shape)
            # x = x.reshape(x.size(0), self.K, x.size(3), x.size(4))

        # print("CNN shape:",x.shape)
        x = self.bn0(x)
        x = self.init_conv(x)

        if self.standard:
            x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        count = 0
        for t in self.parameters():
            count += t.numel()
        return count
 
