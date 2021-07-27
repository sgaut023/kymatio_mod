import torch

from .alternating_optimization import Optimizer

def optimizerFactory(hybridModel, params):
    """Factory for adam, sgd, and a custom alternating optimizer

    parameters: 
        params -- dict of input parameters
        hybridModel -- the model used during training 
    """
    if params['optim']['alternating']:
        return Optimizer(
                    model=hybridModel.top, scatteringModel=hybridModel.scatteringBase, 
                    optimizer_name=params['optim']['name'], lr=params['optim']['lr'], 
                    weight_decay=params['optim']['weight_decay'], momentum=params['optim']['momentum'], 
                    epoch=params['model']['epoch'], num_phase=params['optim']['phase_num'],
                    phaseEnds=params['optim']['phase_ends'],scattering_max_lr=params['scattering']['max_lr'],
                    scattering_div_factor=params['scattering']['div_factor'],scattering_three_phase = params['scattering']['three_phase']
                )

    if params['optim']['name'] == 'adam':
        return torch.optim.Adam(
            hybridModel.parameters(),lr=params['optim']['lr'], 
            betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=params['optim']['weight_decay'], amsgrad=False
        )
    elif params['optim']['name'] == 'sgd': 
        return torch.optim.SGD(
            hybridModel.parameters(), lr=params['optim']['lr'], 
            momentum=params['optim']['momentum'], weight_decay=params['optim']['weight_decay']
        )
        
    else:
        raise NotImplemented(f"Optimizer {params['optim']['name']} not implemented")

