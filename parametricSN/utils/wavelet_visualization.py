import matplotlib.pyplot as plt
import numpy as np

def get_filters_visualization(psi, num_row = 2 , num_col =8 , mode ='fourier'):
    '''
        Function that logs the metrics on MLFLOW
        Params: 
        filters: psi filters
        num_row: number of rows in the visualization
        num_col: number of columns in the visualization
        mode: fourier, real or imag
    '''
    f, axarr = plt.subplots(num_row,num_col, figsize=(20, 5))
    count = 0
    for i in range(0, num_row) :
        for j in range(0, num_col) :
            if mode =='fourier':
                x =np.fft.fftshift(psi[count][0].squeeze().cpu().detach().numpy()).real
            elif mode == 'real':
                x= np.fft.fftshift(np.fft.ifft2(psi[count][0].squeeze().cpu().detach().numpy())).real
            elif mode == 'imag':
                x= np.fft.fftshift(np.fft.ifft2(psi[count][0].squeeze().cpu().detach().numpy())).imag
            else:
                raise NotImplemented(f"Model {params['name']} not implemented")
            axarr[i,j].imshow(x)
            axarr[i,j].set_title(f"J:{psi[count]['j']} L: {psi[count]['theta']}")
            axarr[i,j].axis('off')
            count = count +1
            axarr[i,j].set_xticklabels([])
            axarr[i,j].set_yticklabels([])
            axarr[i,j].set_aspect('equal')

    f.subplots_adjust(wspace=0, hspace=0.2)
    return f