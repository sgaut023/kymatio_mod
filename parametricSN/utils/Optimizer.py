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
    def __init__(self, model, params_filters, optimizer_name, is_scattering_dif , lr, 
                lr_scattering, lr_orientation, weight_decay, momentum, epoch, num_phase=2):
       
        self.model = model
        self.params_filters =  params_filters
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.lr_scattering = lr_scattering
        self.weight_decay = weight_decay
        self.lr_orientation = lr_orientation
        self.is_scattering_dif  = is_scattering_dif 
        self.momentum = momentum
        epoch_phase = math.ceil(epoch/num_phase)  # the number of ecpohs per phase

        # the number of epoch after which we alternate the training between the model and params scattering
        if is_scattering_dif:
            self.epoch_alternate = np.arange(0, epoch, epoch_phase)[0:num_phase]
            self.epoch_alternate = np.delete(self.epoch_alternate,0)
        else:
            self.epoch_alternate = np.array([epoch])
        
        self.phase = 0
        self.define_optimizer(self.model.parameters())
        self.scheduler = None

    def define_optimizer(self, params):
                # the first phae
        if  self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params,lr=self.lr, 
                        betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=self.lr, 
                                         momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            print("Invalid optimizer parameter passed")
    

    def step(self, epoch):
        self.optimizer.step()
        if self.scheduler.scheduler != None:
            self.scheduler.scheduler.step()
        # if we are 
        if epoch in self.epoch_alternate and self.is_scattering_dif:
            self.phase +=1 
            self.epoch_alternate = np.delete(self.epoch_alternate, np.where(self.epoch_alternate==epoch))
            if self.phase %2 ==0:
                params = self.model.parameters()
            else: 
                params =  [{'params': self.params_filters[0], 'lr': self.lr_orientation},
                            {'params': [self.params_filters[1], self.params_filters[2],
                            self.params_filters[3]],'lr': self.lr_scattering} ]
            
            self.define_optimizer(params)
            self.scheduler.define_scheduler()
    
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
        
        # number of iteration to decrease the lr for step lr
        self.step_size = int((self.epochs * self.steps_per_epoch) /num_step)
        self.define_scheduler()


    def define_scheduler(self):
        if self.scheduler_name =='OneCycleLR':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer.optimizer, max_lr=self.max_lr, 
                                                            steps_per_epoch=self.steps_per_epoch, 
                                                            epochs= self.epochs, 
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




        

