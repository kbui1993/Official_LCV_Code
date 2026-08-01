"""
Copyright (c) 2025 Kevin Bui <kevinb3@uci.edu> and Adina Ciomaga <adina@math.univ-paris-diderot.fr>
 
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.
 
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.
 
You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
from scipy.ndimage import gaussian_filter
from compute_energy import *
from utils import *



def localtwophase(f, lamba, beta, dt):
    """
    This function performs local Chan-Vese two-phase segmentation onto a grayscale image.
    Reference: An efficient local Chan & Vese model for image segmentation by Wang et. al
    Input:
        f - original image
        lamba - weighing parameter for the fidelity term
        beta - weighing parameter for the intensity difference term
        dt - time step
    Output:
        u - segmentation result
    """
    # padding parameter
    pad=3

    # creates low-pass filters
    M,N = f.shape
    XX,YY=np.meshgrid(np.linspace(1,N,N), np.linspace(1,M,M))
    XX = XX/N
    YY = YY/M
    freqs_1 = XX - 0.5 - 1/N/2;
    freqs_2 = YY - 0.5 - 1/M/2;
    Lfilter = 1 + dt*(freqs_1**2 + freqs_2**2)

    # creates pre-initialization functions for segmentation (checkerboard functions)
    u = np.kron(np.sin(np.pi*np.linspace(1,M,M)/5), np.transpose(np.sin(np.pi*np.linspace(1,N,N)/5)))
    u = np.reshape(u,[M,N])
    u = np.double(u>0)

    # padding
    f=np.pad(f, pad, 'edge')
    u=np.pad(u, pad, 'edge')
    Lfilter = np.pad(Lfilter, pad, 'edge')

    # create a matrix of small values
    error = (1e-5)*np.eye(u.shape[0], u.shape[1])
    error = error

    # Filtered image
    filtf = gaussian_filter(f,sigma = 5, mode = 'nearest', radius = 10);

    # difference between filtered image and original image
    diff = filtf - f

    # perform two-phase image segmentation
    for i in range(0,200):
        
        # Computes mean-value intensities in each region
        c1=np.sum(f*(u==1))/np.sum((u==1)+error)
        c2=np.sum(f*(u==0))/np.sum((u==0)+error)

        # compute the mean value intensity difference in each region
        d1=np.sum(diff*(u==1))/np.sum((u==1)+error)
        d2=np.sum(diff*(u==0))/np.sum((u==0)+error)

        # computing u
        u_old = u;
        u=u-dt*lamba*((f-c1)**2-(f-c2)**2)
        u=u-dt*beta*((diff-d1)**2-(diff-d2)**2)
        u=np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u))/Lfilter)))

        # Thresholding u
        u = np.double(u >0.5)
        # stopping criterion
        if np.linalg.norm(u-u_old,'fro') < 1e-5:
            break
    
    # compute energy
    energy = grayscale_two_phase_cv_energy(f, diff, u, lamba, beta)

    # print number of iterations to complete convergence
    print(f'Number of iterations completed: {i} \n')
    print(f'Energy: {energy} \n \n')

    # unpadding
    u = u[pad:-pad,pad:-pad]

    return(u)

def localtwophase_color(f, lamba, beta, dt):
    """
    This function performs local Chan-Vese two-phase segmentation onto a color image.
    Input:
        f - original image
        lamba - weighing parameter for the fidelity term
        beta - weighing parameter for the intensity difference term
        dt - time step
    Output:
        u - segmentation result
    """
    # padding parameter
    pad=3

    # creates low-pass filters
    M,N,_ = f.shape
    XX,YY=np.meshgrid(np.linspace(1,N,N), np.linspace(1,M,M))
    XX = XX/N
    YY = YY/M
    freqs_1 = XX - 0.5 - 1/N/2;
    freqs_2 = YY - 0.5 - 1/M/2;
    Lfilter = 1 + dt*(freqs_1**2 + freqs_2**2)

    # split the channel
    f1 = f[:,:,0]
    f2 = f[:,:,1]
    f3 = f[:,:,2]

    # creates pre-initialization functions for segmentation (checkerboard functions)
    u = np.kron(np.sin(np.pi*np.linspace(1,M,M)/5), np.transpose(np.sin(np.pi*np.linspace(1,N,N)/5)))
    u = np.reshape(u,[M,N])
    u = np.double(u>0)

    # padding
    f1=np.pad(f1, pad, 'edge')
    f2=np.pad(f2, pad, 'edge')
    f3=np.pad(f3, pad, 'edge')
    u=np.pad(u, pad, 'edge')
    Lfilter = np.pad(Lfilter, pad, 'edge')

    # create a matrix of small values
    error = (1e-5)*np.eye(u.shape[0], u.shape[1])
    error = error

    # Filtered image
    filtf1 = gaussian_filter(f1,sigma = 5, mode = 'nearest', radius = 10);
    filtf2 = gaussian_filter(f2,sigma = 5, mode = 'nearest', radius = 10);
    filtf3 = gaussian_filter(f3,sigma = 5, mode = 'nearest', radius = 10);


    # difference between filtered image and original image
    diff1 = filtf1 - f1
    diff2 = filtf2 - f2
    diff3 = filtf3 - f3

    # concatenate
    color_f = np.stack([f1,f2,f3],axis = 0)
    color_diff = np.stack([diff1, diff2, diff3], axis=0)


    # perform two-phase image segmentation
    for i in range(0,200):
        
        # Computes mean-value intensities in each region and L2 fidelities for image
        c11=np.sum(f1*(u==1))/np.sum((u==1)+error)
        c12=np.sum(f2*(u==1))/np.sum((u==1)+error)
        c13=np.sum(f3*(u==1))/np.sum((u==1)+error)
        f_minus_c1 = (f1-c11)**2+(f2-c12)**2 + (f3-c13)**2

        c21=np.sum(f1*(u==0))/np.sum((u==0)+error)
        c22=np.sum(f2*(u==0))/np.sum((u==0)+error)
        c23=np.sum(f3*(u==0))/np.sum((u==0)+error)
        f_minus_c2 = (f1-c21)**2+(f2-c22)**2+(f3-c23)**2

        # compute the mean value intensity difference in each region and L2 fidelities for image diff
        d11=np.sum(diff1*(u==1))/np.sum((u==1)+error)
        d12=np.sum(diff2*(u==1))/np.sum((u==1)+error)
        d13=np.sum(diff3*(u==1))/np.sum((u==1)+error)
        diff_minus_d1 = (diff1-d11)**2+(diff2-d12)**2+(diff3-d13)**2
    
        d21=np.sum(diff1*(u==0))/np.sum((u==0)+error)
        d22=np.sum(diff2*(u==0))/np.sum((u==0)+error)
        d23=np.sum(diff3*(u==0))/np.sum((u==0)+error)
        diff_minus_d2 = (diff1-d21)**2+(diff2-d22)**2+(diff3-d23)**2

        # computing u
        u_old = u;
        u=u-dt*lamba*(f_minus_c1-f_minus_c2)
        u=u-dt*beta*(diff_minus_d1-diff_minus_d2)
        u=np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u))/Lfilter)))

        # Thresholding u
        u = np.double(u >0.5)
        # stopping criterion
        if np.linalg.norm(u-u_old,'fro') < 1e-5:
            break
    


    # print number of iterations to complete convergence
    print(f'Number of iterations completed: {i+1} \n')

        # compute energy
    energy = color_two_phase_cv_energy(color_f, color_diff, u, lamba, beta)
    print(f'Energy: {energy} \n\n')

    # unpadding
    u = u[pad:-pad,pad:-pad]

    return(u)

def localfourphase(f, lamba, beta, dt):
    """
    This function performs local Chan-Vese four-phase segmentation onto a grayscale image.
    Input:
        f - original image
        lamba - weighing parameter for the fidelity term
        beta - weighing parameter for the intensity difference term
        dt - time step
    Output:
        u - segmentation result
    """
    # padding parameter
    pad=3

    # creates low-pass filters
    M,N = f.shape
    XX,YY=np.meshgrid(np.linspace(1,N,N), np.linspace(1,M,M))
    XX = XX/N
    YY = YY/M
    freqs_1 = XX - 0.5 - 1/N/2;
    freqs_2 = YY - 0.5 - 1/M/2;
    Lfilter = 1 + dt*(freqs_1**2 + freqs_2**2)

    # creates pre-initialization functions for segmentation (checkerboard functions)
    u1 = np.kron(np.sin(np.pi*np.linspace(1,M,M)/3), np.transpose(np.sin(np.pi*np.linspace(1,N,N)/3)))
    u1 = np.reshape(u1,[M,N])
    u1 = np.double(u1>0)

    u2 = np.kron(np.sin(np.pi*np.linspace(1,M,M)/10), np.transpose(np.sin(np.pi*np.linspace(1,N,N)/10)))
    u2 = np.reshape(u2,[M,N])
    u2 = np.double(u2>0)


    # padding
    f=np.pad(f, pad, 'edge')
    u1=np.pad(u1, pad, 'edge')
    u2=np.pad(u2, pad, 'edge')
    Lfilter = np.pad(Lfilter, pad, 'edge')

    # create a matrix of small values
    error = (1e-5)*np.eye(u1.shape[0], u1.shape[1])
    error = error

    # Filtered image
    filtf = gaussian_filter(f,sigma = 5, mode = 'nearest', radius = 10);

    # difference between filtered image and original image
    diff = filtf - f


    # perform four-phase image segmentation
    for i in range(0,200):
        
        # Computes mean-value intensities in each region
        c1=np.sum(f*(u1==1)*(u2==1))/np.sum((u1==1)*(u2==1)+error)
        c2=np.sum(f*(u1==0)*(u2==1))/np.sum((u1==0)*(u2==1)+error)
        c3=np.sum(f*(u1==1)*(u2==0))/np.sum((u1==1)*(u2==0)+error)
        c4=np.sum(f*(u1==0)*(u2==0))/np.sum((u1==0)*(u2==0)+error)

        # Computes mean-value intensities in each region
        d1=np.sum(diff*(u1==1)*(u2==1))/np.sum((u1==1)*(u2==1)+error)
        d2=np.sum(diff*(u1==0)*(u2==1))/np.sum((u1==0)*(u2==1)+error)
        d3=np.sum(diff*(u1==1)*(u2==0))/np.sum((u1==1)*(u2==0)+error)
        d4=np.sum(diff*(u1==0)*(u2==0))/np.sum((u1==0)*(u2==0)+error)


        # computing u1
        u1_old = u1;
        u1=u1-dt*lamba*(((f-c1)**2)*u2-((f-c2)**2)*u2+((f-c3)**2)*(1-u2)-((f-c4)**2)*(1-u2));
        u1=u1-dt*beta*(((diff-d1)**2)*u2-((diff-d2)**2)*u2+((diff-d3)**2)*(1-u2)-((diff-d4)**2)*(1-u2));
        u1=np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u1))/Lfilter)))

        # Thresholding u1
        u1 = np.double(u1 >0.5)

        # computing u2
        u2_old = u2;
        u2=u2-dt*lamba*(((f-c1)**2)*u1-((f-c3)**2)*u1+((f-c2)**2)*(1-u1)-((f-c4)**2)*(1-u1))
        u2=u2-dt*beta*(((diff-d1)**2)*u1 - ((diff-d3)**2)*u1 + ((diff-d2)**2)*(1-u1)-((diff-d4)**2)*(1-u1))
        u2=np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u2))/Lfilter)))

        # Thresholding u2
        u2 = np.double(u2 >0.5)

        # stopping criterion
        if np.linalg.norm(np.concatenate((u1,u2))-np.concatenate((u1_old,u2_old)),'fro') < 1e-5:
            break
    # print number of iterations to complete convergence
    print(f'Number of iterations completed: {(i+1)} \n')

    # compute energy
    energy = grayscale_four_phase_cv_energy(f,diff, u1, u2, lamba, beta)
    print(f'Energy: {energy} \n\n')

    # unpadding
    u1 = u1[pad:-pad,pad:-pad]
    u2 = u2[pad:-pad,pad:-pad]

    # combining u1 and u2 to obtain final segmentation result
    u1=2*u1
    u2=1*u2
    u=u1+u2

    return(u)

def localfourphase_color(f, lamba, beta, dt):
    """
    This function performs local Chan-Vese four-phase segmentation onto a color image.
    Input:
        f - original image
        lamba - weighing parameter for the fidelity term
        beta - weighing parameter for the intensity difference term
        dt - time step
    Output:
        u - segmentation result
    """
    # padding parameter
    pad=3

    # creates low-pass filters
    M,N,_ = f.shape
    XX,YY=np.meshgrid(np.linspace(1,N,N), np.linspace(1,M,M))
    XX = XX/N
    YY = YY/M
    freqs_1 = XX - 0.5 - 1/N/2;
    freqs_2 = YY - 0.5 - 1/M/2;
    Lfilter = 1 + dt*(freqs_1**2 + freqs_2**2)

    # split the channel
    f1 = f[:,:,0];
    f2 = f[:,:,1];
    f3 = f[:,:,2];

    # creates pre-initialization functions for segmentation (checkerboard functions)
    u1 = np.kron(np.sin(np.pi*np.linspace(1,M,M)/3), np.transpose(np.sin(np.pi*np.linspace(1,N,N)/3)))
    u1 = np.reshape(u1,[M,N])
    u1 = np.double(u1>0)

    u2 = np.kron(np.sin(np.pi*np.linspace(1,M,M)/10), np.transpose(np.sin(np.pi*np.linspace(1,N,N)/10)))
    u2 = np.reshape(u2,[M,N])
    u2 = np.double(u2>0)

    # padding
    f1=np.pad(f1, pad, 'edge')
    f2=np.pad(f2, pad, 'edge')
    f3=np.pad(f3, pad, 'edge')
    u1=np.pad(u1, pad, 'edge')
    u2=np.pad(u2, pad, 'edge')
    Lfilter = np.pad(Lfilter, pad, 'edge')

    # create a matrix of small values
    error = (1e-5)*np.eye(u1.shape[0], u1.shape[1])
    error = error

    # Filtered image
    filtf1 = gaussian_filter(f1,sigma = 5, mode = 'nearest', radius = 10);
    filtf2 = gaussian_filter(f2,sigma = 5, mode = 'nearest', radius = 10);
    filtf3 = gaussian_filter(f3,sigma = 5, mode = 'nearest', radius = 10);

    # difference between filtered image and original image
    diff1 = filtf1 - f1
    diff2 = filtf2 - f2
    diff3 = filtf3 - f3

    # concatenate
    color_f = np.stack([f1,f2,f3],axis = 0)
    color_diff = np.stack([diff1, diff2, diff3], axis=0)


    # perform four-phase image segmentation
    for i in range(0,200):
        
        # Computes mean-value intensities in each region and L2 fidelities for image
        c11=np.sum(f1*(u1==1)*(u2==1))/np.sum((u1==1)*(u2==1)+error)
        c12=np.sum(f2*(u1==1)*(u2==1))/np.sum((u1==1)*(u2==1)+error)
        c13=np.sum(f3*(u1==1)*(u2==1))/np.sum((u1==1)*(u2==1)+error)
        f_minus_c1 = (f1-c11)**2 + (f2-c12)**2 + (f3-c13)**2

        c21=np.sum(f1*(u1==0)*(u2==1))/np.sum((u1==0)*(u2==1)+error)
        c22=np.sum(f2*(u1==0)*(u2==1))/np.sum((u1==0)*(u2==1)+error)
        c23=np.sum(f3*(u1==0)*(u2==1))/np.sum((u1==0)*(u2==1)+error)
        f_minus_c2 = (f1-c21)**2 + (f2-c22)**2 + (f3-c23)**2

        c31=np.sum(f1*(u1==1)*(u2==0))/np.sum((u1==1)*(u2==0)+error)
        c32=np.sum(f2*(u1==1)*(u2==0))/np.sum((u1==1)*(u2==0)+error)
        c33=np.sum(f3*(u1==1)*(u2==0))/np.sum((u1==1)*(u2==0)+error)
        f_minus_c3 = (f1-c31)**2 + (f2-c32)**2 + (f3-c33)**2

        
        c41=np.sum(f1*(u1==0)*(u2==0))/np.sum((u1==0)*(u2==0)+error)
        c42=np.sum(f2*(u1==0)*(u2==0))/np.sum((u1==0)*(u2==0)+error)
        c43=np.sum(f3*(u1==0)*(u2==0))/np.sum((u1==0)*(u2==0)+error)
        f_minus_c4 = (f1-c41)**2 + (f2-c42)**2 + (f3-c43)**2

        # Computes mean-value intensities in each region and L2 fidelitie for image difference
        d11=np.sum(diff1*(u1==1)*(u2==1))/np.sum((u1==1)*(u2==1)+error)
        d12=np.sum(diff2*(u1==1)*(u2==1))/np.sum((u1==1)*(u2==1)+error)
        d13=np.sum(diff3*(u1==1)*(u2==1))/np.sum((u1==1)*(u2==1)+error)
        diff_minus_d1 = (diff1-d11)**2 + (diff2-d12)**2 + (diff3-d13)**2

        d21=np.sum(diff1*(u1==0)*(u2==1))/np.sum((u1==0)*(u2==1)+error)
        d22=np.sum(diff2*(u1==0)*(u2==1))/np.sum((u1==0)*(u2==1)+error)
        d23=np.sum(diff3*(u1==0)*(u2==1))/np.sum((u1==0)*(u2==1)+error)
        diff_minus_d2 = (diff1-d21)**2 + (diff2-d22)**2 + (diff3-d23)**2

        d31=np.sum(diff1*(u1==1)*(u2==0))/np.sum((u1==1)*(u2==0)+error)
        d32=np.sum(diff2*(u1==1)*(u2==0))/np.sum((u1==1)*(u2==0)+error)
        d33=np.sum(diff3*(u1==1)*(u2==0))/np.sum((u1==1)*(u2==0)+error)
        diff_minus_d3 = (diff1-d31)**2 + (diff2-d32)**2 + (diff3-d33)**2

        d41=np.sum(diff1*(u1==0)*(u2==0))/np.sum((u1==0)*(u2==0)+error)
        d42=np.sum(diff2*(u1==0)*(u2==0))/np.sum((u1==0)*(u2==0)+error)
        d43=np.sum(diff3*(u1==0)*(u2==0))/np.sum((u1==0)*(u2==0)+error)
        diff_minus_d4 = (diff1-d41)**2 + (diff2-d42)**2 + (diff3-d43)**2


        # computing u1
        u1_old = u1
        u1=u1-dt*lamba*(f_minus_c1*u2-f_minus_c2*u2+f_minus_c3*(1-u2)-f_minus_c4*(1-u2))
        u1=u1-dt*beta*(diff_minus_d1*u2-diff_minus_d2*u2+diff_minus_d3*(1-u2)-diff_minus_d4*(1-u2))
        u1=np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u1))/Lfilter)))

        # Thresholding u1
        u1 = np.double(u1 >0.5)

        # computing u2
        u2_old = u2;
        u2=u2-dt*lamba*(f_minus_c1*u1-f_minus_c3*u1+f_minus_c2*(1-u1)-f_minus_c4*(1-u1))
        u2=u2-dt*beta*(diff_minus_d1*u1 - diff_minus_d3*u1 + diff_minus_d2*(1-u1)-diff_minus_d4*(1-u1))
        u2=np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u2))/Lfilter)))

        # Thresholding u2
        u2 = np.double(u2 >0.5)

        # stopping criterion
        if np.linalg.norm(np.concatenate((u1,u2))-np.concatenate((u1_old,u2_old)),'fro') < 1e-5:
            break
    # print number of iterations to complete convergence
    print(f'Number of iterations completed: {(i+1)} \n ')

    energy = color_four_phase_cv_energy(color_f,color_diff, u1, u2, lamba, beta)
    print(f'Energy: {energy} \n\n')


    # unpadding
    u1 = u1[pad:-pad,pad:-pad]
    u2 = u2[pad:-pad,pad:-pad]

    # combining u1 and u2 to obtain final segmentation result
    u1=2*u1
    u2=1*u2
    u=u1+u2

    return(u)