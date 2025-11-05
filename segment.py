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
import argparse
from PIL import Image

from lcv import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='./images/grayscale_two_phase/vessel2.png', type=str, metavar='PATH',
                    help='path to the image to be segmented')
parser.add_argument('--lamba', default = 50, type = float, 
                    help = 'weighing parameter for the fidelity term')
parser.add_argument('--beta', default = 500, type = float, 
                    help = 'weighing parameter for the intensity difference term')
parser.add_argument('--dt', default = 20, type = float, 
                    help = 'time step')
parser.add_argument('--phase', default = 'two', type = str,
                    help = "number of phases: 'two' - 2, 'four' - 4")
parser.add_argument('--save', default = './output/output.png', type = str, metavar='PATH',
                    help='file name to save output image as')
args = parser.parse_args()

assert args.phase in ['two', 'four']

def main():
    orig_f = Image.open(args.image)
    orig_f = np.array(orig_f)

    if len(orig_f.shape) == 3:
        f = rgb2lab(orig_f)
        f = rescale_color_image(f)

        if args.phase == 'two':
            u = localtwophase_color(f, args.lamba, args.beta, args.dt)
        elif args.phase == 'four':
            u = localfourphase_color(f, args.lamba, args.beta, args.dt)

        approx = reconstruct_color_image(orig_f, u)
        output = Image.fromarray(np.uint8(approx))
        output.save(args.save)
    elif len(orig_f.shape) == 2:
        
        f = rescale_image(orig_f)

        if args.phase == 'two':
            u = localtwophase(f, args.lamba, args.beta, args.dt)
        elif args.phase == 'four':
            u = localfourphase(f, args.lamba, args.beta, args.dt)
    
        approx = reconstruct_grayscale_image(orig_f, u)
        
        output = Image.fromarray(np.uint8(approx))
        output.save(args.save)

if __name__ == '__main__':
    main()