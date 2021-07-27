"""Contains the factories for selecting different models

Author: Benjamin Therien

functions: 
    baseModelFactory -- Factory for the creation of the first part of a hybrid model
    topModelFactory -- Factory for the creation of seconds part of a hybrid model
"""


from .sn_base_models import sn_Identity, sn_ScatteringBase
from .sn_top_models import sn_CNN, sn_MLP, sn_LinearLayer, sn_Resnet50
from .sn_models_exceptions import InvalidArchitectureError

def baseModelFactory(architecture, J, N, M, second_order, initialization, seed, device, 
                     learnable=True, lr_orientation=0.1,lr_scattering=0.1,use_cuda=True):
    """Factory for the creation of the first layer of a hybrid model
    
    parameters:
        architecture -- the name of the top model to select
        J -- 
        N -- 
        M -- 
        second_order -- 
        initialization -- 
        seed -- 
        device -- 
        learnable=True
        lr_orientation -- 
        lr_scattering -- 
        use_cuda -- boolean indicating whether to use GPU
    """

    if architecture.lower() == 'identity':
        return sn_Identity()

    elif architecture.lower() == 'scattering':
        return sn_ScatteringBase( #create learnable of non-learnable scattering
            J=J,
            N=N,
            M=M,
            second_order=second_order,
            initialization=initialization,
            seed=seed,
            learnable=learnable,
            lr_orientation=lr_orientation,
            lr_scattering=lr_scattering,
            device=device ,
            use_cuda=use_cuda
        )

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()



def topModelFactory(base, architecture, num_classes, width=8, average=False, use_cuda=True):
    """Factory for the creation of seconds part of a hybrid model
    
    parameters:
        base -- (Pytorch nn.Module) the first part of a hybrid model
        architecture -- the name of the top model to select
        num_classes -- number of classes in dataset
        width -- the width of the model
        average -- (TODO Shanel)
        use_cuda -- boolean indicating whether to use GPU
    """

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

    elif architecture.lower() == 'resnet50':
        return sn_Resnet50(num_classes=num_classes)

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()