"""
Copyright (c) 2025 Kevin Bui <kevinb3@uci.edu> and Adina Ciomaga <adina@math.univ-paris-diderot.fr>
 
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.
 
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.
 
You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np

from skimage.color.colorconv import _prepare_colorarray
from skimage.color import xyz2lab

def rgb2xyz(rgb):
    """
    This function maps an rgb image to xyz color space.
    Input:
       rgb: rgb image
    Output:
        result: xyz image
    """
    xyz_from_rgb = np.array([[0.412456439089692,   0.357576077643909,   0.180437483266399],
                         [0.212672851405623,   0.715152155287818,   0.072174993306560],
                         [0.019333895582329,   0.119192025881303,   0.950304078536368]])
    arr = _prepare_colorarray(rgb, channel_axis=-1).copy()
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    result = arr @ xyz_from_rgb.T.astype(arr.dtype)
    return result

def rgb2lab(rgb):
    """
    This function maps an rgb image to Lab color space.
    Input:
       rgb: rgb image
    Output:
        result: Lab image
    """
    result = xyz2lab(rgb2xyz(rgb))
    return result

def rescale_image(F):
    """
    This function rescales an image so that the image values are between 0 and 1.
    Input:
       F: image to be rescaled
    Output:
       h: rescaled image F
    """
    h = np.double(F)
    h= h-np.min(h)
    h = h/np.max(h)

    return h

def rescale_color_image(F):
    """
    This function rescales a color image so that the each channel values are 
    between 0 and 1.
    Input:
       F: color image to be rescaled
    Output:
        h: rescaled image F
    """
    h = F
    for i in range(0,3):
        h[:,:,i] = rescale_image(F[:,:,i])
    return h

def reconstruct_grayscale_image(f, segment_result):
    """
    This function reconstructs the piecewise-constant approximation of
    a grayscale image based on its segmentation result.
    Input:
        f - original grayscale image
        segment_result - segmentation result produced from localtwophase.m or
                         localmultiphase.m. It should contain only integer values: 0,1 for
                         two-phase 0,1,2,3 for 4-phase.
    Output:
        pwc_result - piecewise-constant approximation of the original image f
    """
    # convert to double
    f = np.double(f)

    # count the number of regions
    num_label = len(np.unique(segment_result))

    if num_label == 2:
        c1 = np.sum(f*(segment_result==0))/np.sum(segment_result==0)
        c2 = np.sum(f*(segment_result==1))/np.sum(segment_result==1)
        pwc_result = c1*(segment_result ==0) + c2*(segment_result==1)
    elif num_label == 4:
        c1 = np.sum(f*(segment_result==0))/np.sum(segment_result==0)
        c2 = np.sum(f*(segment_result==1))/np.sum(segment_result==1)
        c3 = np.sum(f*(segment_result==2))/np.sum(segment_result==2)
        c4 = np.sum(f*(segment_result==3))/np.sum(segment_result==3)
        
        pwc_result = c1*(segment_result==0) + c2*(segment_result==1)+c3*(segment_result==2)+c4*(segment_result==3)
    
    return(pwc_result)
        
def reconstruct_color_image(f, segment_result):
    """
    This function reconstructs the piecewise-constant approximation of
    a colorimage based on its segmentation result.
    Input:
        f - original color image
        segment_result - segmentation result produced from localtwophase.m or
                         localmultiphase.m. It should contain only integer values: 0,1 for
                         two-phase 0,1,2,3 for 4-phase.
    Output:
        pwc_result - piecewise-constant approximation of the original image f
    """
    # convert to double
    f = np.double(f)

    # pre-initialize pwc_result
    pwc_result = np.zeros(f.shape)

    # apply reconstruct_grayscale_image to each channel of f
    pwc_result[:,:,0] = reconstruct_grayscale_image(f[:,:,0], segment_result)
    pwc_result[:,:,1] = reconstruct_grayscale_image(f[:,:,1], segment_result)
    pwc_result[:,:,2] = reconstruct_grayscale_image(f[:,:,2], segment_result)
    
    return(pwc_result)

