import torch
import numpy as np
import math

class Optimizer():
    '''
        Class that is responsible to manage the optimization of the scattering network
        and the optimizaiton of the model on top of the scattering network.
        The first phase of the training is always the training of of the model on top.
        If we are optimizing the scattering params then:
        The second phase is the training of the scattering params.
        When the phase number is even, we are training the model.
        When the pahse number is odd, we are training the scattering params.
    '''
    def __init__(self, model, scatteringModel, optimizer_name , lr, 
                 weight_decay, momentum, epoch, num_phase=2):

        self.model = model
        self.scatteringModel = scatteringModel
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        epoch_phase = math.ceil(epoch/num_phase)  # the number of ecpohs per phase

        # the number of epoch after which we alternate the training between the model and params scattering
        if self.scatteringModel.learnable:
            self.epoch_alternate = np.arange(0, epoch, epoch_phase)[0:num_phase]
            self.epoch_alternate = np.delete(self.epoch_alternate,0)
        else:
            self.epoch_alternate = np.array([epoch])

        self.phase = 0
        self.define_optimizer(self.model.parameters())
        self.param_groups = self.optimizer.param_groups
        self.scheduler = None

    def define_optimizer(self, parameters):
        if  self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(parameters,lr=self.lr, 
                        betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, lr=self.lr, 
                                         momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise NotImplemented(f"Optimizer {self.optimizer_name} not implemented")

        self.param_groups = self.optimizer.param_groups


    def zero_grad(self):
        self.optimizer.zero_grad()


    def step(self, epoch):
        self.optimizer.step()
        # if self.scheduler.scheduler != None:
        #     self.scheduler.scheduler.step()

        #create new optimizer and 
        if epoch in self.epoch_alternate and self.scatteringModel.learnable:
            self.phase +=1 
            self.epoch_alternate = np.delete(self.epoch_alternate, np.where(self.epoch_alternate==epoch))
            if self.phase %2 ==0:
                params = self.model.parameters()
            else: 
                params = self.scatteringModel.parameters()

            self.define_optimizer(params)
            self.scheduler.define_scheduler()
            self.scheduler.skipStep = True #skip the step to make this compatible with scheduler trianing loop behaviours

class Scheduler():
    def __init__(self, optimizer, scheduler_name, steps_per_epoch, epochs, div_factor= 25, 
                 max_lr =0.05, T_max = None, num_step = 3):
        self.optimizer = optimizer
        self.scheduler_name = scheduler_name
        self.max_lr = max_lr
        self.div_factor = div_factor
        self.steps_per_epoch = steps_per_epoch
        self.epochs = int(epochs)
        self.T_max = T_max
        self.num_step = num_step
        self.scheduler = None
        self.skipStep = False

        # number of iteration to decrease the lr for step lr
        self.step_size = int((self.epochs * self.steps_per_epoch) /num_step)
        self.define_scheduler()

    def step(self):
        if self.skipStep:
            self.skipStep = False
        else:
            self.scheduler.step()


    def define_scheduler(self):
        if self.scheduler_name =='OneCycleLR':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer.optimizer, max_lr=self.max_lr, 
                                                                 steps_per_epoch=self.steps_per_epoch, 
                                                                 epochs=self.epochs+1, 
                                                                 three_phase=False,
                                                                 div_factor=self.div_factor)
        elif self.scheduler_name =='CosineAnnealingLR':
            self.scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer.optimizer, T_max = self.T_max, eta_min = 1e-8)
        elif self.scheduler_name =='LambdaLR':
            lmbda = lambda epoch: 0.95
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer.optimizer, lr_lambda=lmbda)
        elif self.scheduler_name =='StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer.optimizer, step_size=self.step_size, gamma=0.2)
        elif self.scheduler_name == 'NoScheduler':
            self.scheduler = None
        else:
            raise NotImplemented(f"Scheduler {self.scheduler_name} not implemented")



