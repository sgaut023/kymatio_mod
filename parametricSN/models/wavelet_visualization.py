"""Contains all the funcitons related to visualizing the different scattering filters

Functions:  
    get_filters_visualization -- 

"""

import matplotlib.pyplot as plt
import numpy as np

def get_filters_visualization(psi, J, L, mode ='fourier'):
    """visualizes the scattering filters input for different modes

    TODO Shanel

    parameters:
        psi -- 
        L -- 
        J -- 
        mode -- 
    """
    n_filters =0
    for j in range(2, J+1):
        n_filters+=  j* L
    num_rows = int(n_filters/L) 
    num_col = L
    f, axarr = plt.subplots(num_rows, num_col, figsize=(20, 2*num_rows))
    start_row = 0
    for scale in range(J-1):
        count = L * scale
        end_row = (J-scale) + start_row 
        for i in range(start_row, end_row) :
            for j in range(0, L) :
                if mode =='fourier':
                    x =np.fft.fftshift(psi[count][scale].squeeze().cpu().detach().numpy()).real
                elif mode == 'real':
                    x= np.fft.fftshift(np.fft.ifft2(psi[count][scale].squeeze().cpu().detach().numpy())).real
                elif mode == 'imag':
                    x= np.fft.fftshift(np.fft.ifft2(psi[count][scale].squeeze().cpu().detach().numpy())).imag
                else:
                    raise NotImplemented(f"Model {params['name']} not implemented")
                
                a=np.abs(x).max()
                axarr[i,j].imshow(x, vmin=-a, vmax=a)
                axarr[i,j].set_title(f"J:{psi[count]['j']} L: {psi[count]['theta']}, S:{scale} ")
                axarr[i,j].axis('off')
                count = count +1
                axarr[i,j].set_xticklabels([])
                axarr[i,j].set_yticklabels([])
                axarr[i,j].set_aspect('equal')
        start_row = end_row

    f.subplots_adjust(wspace=0, hspace=0.2)
    return f