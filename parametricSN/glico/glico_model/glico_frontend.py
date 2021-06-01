"""FrontEnd to the glico model for data augmentation 


"""

import torch
import time
import itertools
import collections
import torchvision

import numpy as np
import torchvision.utils as vutils

from numpy.random import RandomState
from torch.utils.data import Subset

from newglico.glico_model.nag_trainer import NAGTrainer
from newglico.glico_model.interpolate import interpolate_points

OptParams = collections.namedtuple(
    'OptParams', 
    'lr factor ' 
    + 'batch_size epochs ' 
    + 'decay_epochs decay_rate gamma'
    )
    
OptParams.__new__.__defaults__ = (None, None, None, None, None, None, None)

NAGParams = collections.namedtuple(
    'NAGParams',
    'nz force_l2 is_pixel z_init is_classifier disc_net loss data_name noise_proj shot'
    )

NAGParams.__new__.__defaults__ = (None, None, None, None)


def trainGlico(train_labeled_dataset, numClasses, epochs):
    """Trains the glico generator for data augmentation

    parameters:
        train_labeled_dataset -- training set for the generator
        numClasses -- the number of classes in the dataset
        epochs -- the number of epochs to train for
    """
    FILE_NAME = "newGlicoEra"

    nag_params = NAGParams(nz=512, force_l2=False, is_pixel=True, z_init='rndm', is_classifier=True, #GLICO API
                           disc_net='conv', loss='ce', data_name='cifar-10', noise_proj=True, shot=0)

    nag_opt_params = OptParams(lr=0.001, factor=0.7, batch_size=128, epochs=epochs, #GLICO API
                               decay_epochs=70, decay_rate=0.5, gamma=0.5)

    nag_trainer = NAGTrainer(dataset=[train_labeled_dataset, [], []], nag_params=nag_params, #GLICO API
                             rn=FILE_NAME, resume=False, num_classes=numClasses)

    nag_trainer.train_test_nag(nag_opt_params)

    return nag_trainer


def getGenImage(z1,z2,gen,steps,codeSize):
    """gets interpolated images between z1,z2

    parameters:
        z1 -- latent vec1
        z2 -- latent vec2
        gen -- the generator
        steps -- number of steps to interpolate between z1 and z2
        codeSize -- the size of the noise concatenated to the latent in the forward pass
    """
    interp = interpolate_points(z1, z2, n_steps=steps, slerp=True, print_mode=False)
    code = torch.cuda.FloatTensor(steps, codeSize).normal_(0, 0.15)
    im = gen(interp,code)
    return im


class GlicoLoader:
    """Object to hold the Glico models in order to sample from them to
    obtain new training images
    
    """
    
    def getInterpolations(self):
        """Obtains a specified number of interpolations between images of the same class"""

        interps = [[x for x in itertools.combinations(self.netZ.label2idx[y], 2)] for y in range(10)]
        self.interps = {}
        for i,combinations in enumerate(interps):
            zvecs = self.netZ(torch.tensor(combinations).cuda())
            temp = []
            for idx in range(zvecs.size(0)):
                z1 = zvecs[idx,0,:]
                z2 = zvecs[idx,1,:]
                gen = getGenImage(z1,z2,self.netG,steps=self.steps,codeSize=self.nag.code_size)
                temp.append(gen.detach())
                
            self.interps[i] = torch.cat(temp,dim=0)
            
        
    def __init__(self,nagTrainer,steps):
        """Parse GLICO models out of their objects

        parameters:
            nagTrainer -- glico object holding all the models
            steps -- the number of interpolation steps between image pairs
        """
        self.nagTrainer = nagTrainer
        self.nag = nagTrainer.nag
        self.netZ = self.nag.netZ
        self.netG = self.nag.netG
        self.steps = steps
        self.getInterpolations()
        
        self.indices = [0 for x in range(10)]
        a = self.interps[0].size(0)
        prng = RandomState(int(time.time()))
        self.indexers = [list(prng.permutation(np.arange(0,a))) for x in range(10)]
        
        
    def sample(self,classNum):
        if self.interps[0].size(0) == self.indices[classNum]:
            self.indices[classNum] = 0
            prng = RandomState(int(time.time()))
            self.indexers[classNum] = list(prng.permutation(np.arange(0,self.interps[0].size(0))))
            
        temp = self.interps[classNum][self.indexers[classNum][self.indices[classNum]],:,:,:]
            
        return temp
    
    def replaceBatch(self,batch,targets,replaceProb):
        for x in range(batch.size(0)):
            prob = np.random.rand()
            if prob < replaceProb:
                batch[x,:,:,:] = self.sample(targets[x].item())
            else:
                pass
        return batch

    def visualize(self):

        vutils.save_image(Irec.data, f'runs/ims_{self.rn}/reconstructions_{image_size}__{epoch}_vutils.png',
                          normalize=False)


