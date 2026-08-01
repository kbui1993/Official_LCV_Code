"""
Copyright (c) 2026 Kevin Bui <kevinb3@uci.edu> and Adina Ciomaga <adina@math.univ-paris-diderot.fr>
 
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

def grayscale_two_phase_cv_energy(image, diff, phi, lamba, beta):
    """Returns the total 'energy' of the current level set function for two-phase grayscale image segmentation.
    This corresponds to the (LCV) equation of the paper.
    Input:
        image - original image
        diff - image difference between filtered image and original image
        phi - level set function
        lamba - weighing parameter for the fidelity term
        beta - weighing parameter for the intensity difference term
    Output: eLCV energy for two-phase segmentation for grayscale image
    """
    # compute the Heaviside of the level set function
    H = _cv_heavyside(phi)

    # compute the L2 fidelity term between original image and the segmented image
    avgenergy = _cv_difference_from_average_term(image, H, lamba)

    # compute the local energy term
    localenergy = _cv_difference_from_average_term(diff, H, beta)

    # compute TV or edge length term
    lenenergy = _cv_edge_length_term(phi)

    # return total energy
    return np.sum(avgenergy) + np.sum(localenergy) + np.sum(lenenergy)

def color_two_phase_cv_energy(color_image, color_diff, phi, lamba, beta):
    """Returns the total 'energy' of the current level set function for two-phase color image segmentation.
    This corresponds to the (LCV) equation of the paper.
    Input:
        color_image - original image; shape (3, M, N)
        color_diff - image difference between filtered image and original image; shape (3, M, N)
        phi - level set function
        lamba - weighing parameter for the fidelity term
        beta - weighing parameter for the intensity difference term
    Output: eLCV energy for two-phase segmentation for color image
    """

    # compute the Heaviside function of the level set function
    H = _cv_heavyside(phi)

    # compute the L2 fidelity term between original image and the segmented image
    avgenergy1 = _cv_difference_from_average_term(color_image[0,:,:], H, lamba)
    avgenergy2 = _cv_difference_from_average_term(color_image[1,:,:], H, lamba)
    avgenergy3 = _cv_difference_from_average_term(color_image[2,:,:], H, lamba)
    total_avg = np.sum(avgenergy1+avgenergy2+avgenergy3)

    # compute the local energy term
    localenergy1 = _cv_difference_from_average_term(color_diff[0,:,:], H, beta)
    localenergy2 = _cv_difference_from_average_term(color_diff[1,:,:], H, beta)
    localenergy3 = _cv_difference_from_average_term(color_diff[2,:,:], H, beta)
    total_local_avg = np.sum(localenergy1+localenergy2+localenergy3)

    # compute TV or edge length term
    lenenergy = _cv_edge_length_term(phi)

    # return total energy
    return total_avg + total_local_avg + np.sum(lenenergy)

def grayscale_four_phase_cv_energy(image, diff, phi1, phi2, lamba, beta):
    """Returns the total 'energy' of the current two level set functions for four-phase grayscale image segmentation.
    Input:
        image - original image
        diff - image difference between filtered image and original image
        phi1 - first level set function
        phi2 - second level set function
        lamba - weighing parameter for the fidelity term
        beta - weighing parameter for the intensity difference term
    Output: eLCV energy for four-phase segmentation for grayscale image
    """

    # compute the Heaviside function of the level set functions
    H1 = _cv_heavyside(phi1)
    H2 = _cv_heavyside(phi2)

    # compute the L2 fidelity term between original image and the segmented image
    avgenergy1 = _cv_difference_from_average_term2(image, H1*H2, lamba)
    avgenergy2 = _cv_difference_from_average_term2(image, (1.0-H1)*H2, lamba)
    avgenergy3 = _cv_difference_from_average_term2(image, H1*(1.0-H2), lamba)
    avgenergy4 = _cv_difference_from_average_term2(image, (1.0-H1)*(1.0-H2), lamba)
    total_avg = np.sum(avgenergy1+avgenergy2+avgenergy3+avgenergy4)

    # compute the local energy term
    localenergy1 = _cv_difference_from_average_term2(diff, H1*H2, beta)
    localenergy2 = _cv_difference_from_average_term2(diff, (1.0-H1)*H2, beta)
    localenergy3 = _cv_difference_from_average_term2(diff, H1*(1.0-H2), beta)
    localenergy4 = _cv_difference_from_average_term2(diff, (1-H1)*(1-H2), beta)
    total_local = np.sum(localenergy1+localenergy2+localenergy3+localenergy4)

    # compute TV or edge length term        
    lenenergy1 = _cv_edge_length_term(phi1)
    lenenergy2 = _cv_edge_length_term(phi2)
    total_le = np.sum(lenenergy1+lenenergy2)

    # return total energy       
    return total_avg+total_local+total_le


def color_four_phase_cv_energy(color_image, color_diff, phi1, phi2, lamba, beta):
    """Returns the total 'energy' of the current level set functions for four-phase color image segmentation.
    Input:
        color_image - original image; shape (3, M, N)
        color_diff - image difference between filtered image and original image; shape (3, M, N)
        phi1 - first level set function
        phi2 - second level set function
        lamba - weighing parameter for the fidelity term
        beta - weighing parameter for the intensity difference term
    Output: eLCV energy for four-phase segmentation for color image
    """

    # compute the Heaviside function of the level set functions
    H1 = _cv_heavyside(phi1)
    H2 = _cv_heavyside(phi2)

    # compute the L2 fidelity term between original image and the segmented image
    avgenergy11 = _cv_difference_from_average_term2(color_image[0,:,:], H1*H2, lamba)
    avgenergy12 = _cv_difference_from_average_term2(color_image[1,:,:], H1*H2, lamba)
    avgenergy13 = _cv_difference_from_average_term2(color_image[2,:,:], H1*H2, lamba)
    total_avg1 = np.sum(avgenergy11+avgenergy12+avgenergy13)

    avgenergy21 = _cv_difference_from_average_term2(color_image[0,:,:], (1.0-H1)*H2, lamba)
    avgenergy22 = _cv_difference_from_average_term2(color_image[1,:,:], (1.0-H1)*H2, lamba)
    avgenergy23 = _cv_difference_from_average_term2(color_image[2,:,:], (1.0-H1)*H2, lamba)
    total_avg2 = np.sum(avgenergy21+avgenergy22+avgenergy23)

    avgenergy31 = _cv_difference_from_average_term2(color_image[0,:,:], H1*(1.0-H2), lamba)
    avgenergy32 = _cv_difference_from_average_term2(color_image[1,:,:], H1*(1.0-H2), lamba)
    avgenergy33 = _cv_difference_from_average_term2(color_image[2,:,:], H1*(1.0-H2), lamba)
    total_avg3 = np.sum(avgenergy31+avgenergy32+avgenergy33)

    avgenergy41 = _cv_difference_from_average_term2(color_image[0,:,:], (1.0-H1)*(1.0-H2), lamba)
    avgenergy42 = _cv_difference_from_average_term2(color_image[1,:,:], (1.0-H1)*(1.0-H2), lamba)
    avgenergy43 = _cv_difference_from_average_term2(color_image[2,:,:], (1.0-H1)*(1.0-H2), lamba)
    total_avg4 = np.sum(avgenergy41+avgenergy42+avgenergy43)

    total_avg = total_avg1+total_avg2+total_avg3+total_avg4


    # compute the local energy term  
    localavgenergy11 = _cv_difference_from_average_term2(color_diff[0,:,:], H1*H2, beta)
    localavgenergy12 = _cv_difference_from_average_term2(color_diff[1,:,:], H1*H2, beta)
    localavgenergy13 = _cv_difference_from_average_term2(color_diff[2,:,:], H1*H2, beta)
    total_localavg1 = np.sum(localavgenergy11+localavgenergy12+localavgenergy13)

    localavgenergy21 = _cv_difference_from_average_term2(color_diff[0,:,:], (1.0-H1)*H2, beta)
    localavgenergy22 = _cv_difference_from_average_term2(color_diff[1,:,:], (1.0-H1)*H2, beta)
    localavgenergy23 = _cv_difference_from_average_term2(color_diff[2,:,:], (1.0-H1)*H2, beta)
    total_localavg2 = np.sum(localavgenergy21+localavgenergy22+localavgenergy23)

    localavgenergy31 = _cv_difference_from_average_term2(color_diff[0,:,:], H1*(1.0-H2), beta)
    localavgenergy32 = _cv_difference_from_average_term2(color_diff[1,:,:], H1*(1.0-H2), beta)
    localavgenergy33 = _cv_difference_from_average_term2(color_diff[2,:,:], H1*(1.0-H2), beta)
    total_localavg3 = np.sum(localavgenergy31+localavgenergy32+localavgenergy33)

    localavgenergy41 = _cv_difference_from_average_term2(color_diff[0,:,:], (1.0-H1)*(1.0-H2), beta)
    localavgenergy42 = _cv_difference_from_average_term2(color_diff[1,:,:], (1.0-H1)*(1.0-H2), beta)
    localavgenergy43 = _cv_difference_from_average_term2(color_diff[2,:,:], (1.0-H1)*(1.0-H2), beta)
    total_localavg4 = np.sum(localavgenergy41+localavgenergy42+localavgenergy43)

    total_local = total_localavg1+total_localavg2+total_localavg3+total_localavg4

    # compute TV or edge length term        
    lenenergy1 = _cv_edge_length_term(phi1)
    lenenergy2 = _cv_edge_length_term(phi2)
    total_le = np.sum(lenenergy1+lenenergy2)    
    
    # return total energy
    return total_avg+total_local+total_le

def _cv_calculate_averages(image, Hphi):
    """Returns the average values 'inside' and 'outside'.
    Input:
        image: image input
        Hphi: level set function for segmentation
    Output:
        avg_inside: average intensity value inside the region
        avg_outside: average intensity value outside the region
    """
    H = Hphi
    Hinv = 1.0 - H
    Hsum = np.sum(H)
    Hinvsum = np.sum(Hinv)
    avg_inside = np.sum(image * H)
    avg_oustide = np.sum(image * Hinv)
    if Hsum != 0:
        avg_inside /= Hsum
    if Hinvsum != 0:
        avg_oustide /= Hinvsum
    return (avg_inside, avg_oustide)


def _cv_difference_from_average_term(image, Hphi, lamba):
    """Returns the 'energy' contribution due to the difference from
    the average value within a region at each point. It sums up both
    the energy inside and outside the region.
    Input:
        image: image input
        Hphi: level set function corresponding to a region
        lamba: weighing parameter for the fidelity term
    Output: L2 fidelity energy for both inside and outside a region
    """
    (c1, c2) = _cv_calculate_averages(image, Hphi)
    Hinv = 1.0 - Hphi
    return lamba * (image - c1) ** 2 * Hphi + lamba * (image - c2) ** 2 * Hinv

def _cv_difference_from_average_term2(image, Hphi, lamba):
    """Returns the 'energy' contribution due to the difference from
    the average value inside a region at each point.
    Input:
        image: image input
        Hphi: level set function corresponding to a region
        lamba: weighing parameter for the fidelity term
    Output: L2 fidelity energy for inside a region
    """
    (c1, c2) = _cv_calculate_averages(image, Hphi)
    return lamba * (image - c1) ** 2


def _cv_edge_length_term(phi):
    """Returns the 'energy' contribution due to the length of the
    edge between regions at each point.
    Input:
        phi: level set function corresponding to a region
    Output: total variation energy that is equivalent to the perimeter of the edge
    """
    P = np.pad(phi, 1, mode='edge')
    P = np.double(P>0)
    fy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
    fx = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
    return np.sqrt(fx**2 + fy**2)


def _cv_heavyside(x, eps=1.0):
    """Returns the result of a regularised heavyside function of the
    input value(s).
    Input:
        x: input value
        eps: error term for denominator of Heaviside function
    Output: Heaviside function of x
    """
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(x / eps))