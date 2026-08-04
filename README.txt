================================================================================
## MBO Scheme for Local Chan-Vese Segmenation ##
================================================================================

#ABOUT
------

* Authors   : Kevin Bui (kevinb3@uci.edu) and Adina Ciomaga (adina@math.univ-paris-diderot.fr)
 
Copyright : (C) 2009-2026 IPOL Image Processing On Line http://www.ipol.im/
License   : 

CC Creative Commons
* "Attribution-NonCommercial-ShareAlike" 
see http://creativecommons.org/licenses/by-nc-sa/3.0/es/deed.en

================================================================================
 
# CONTENTS
----------
 
  - Overview
  - Requirements
  - Usage
  - Source Code Organization
  - Thanks
 
================================================================================

# OVERVIEW
----------

This Python source code provides implementation of the local Chan-Vese 
segmentation for two-phase and four-phase segmentation. The implementation is
applicable to grayscale and RGB images. The source code accompanies the 
Image Processing On Line (IPOL) article "MBO Scheme for Local Chan-Vese 
Segmentation" at 

    *website link*

================================================================================
 
# REQUIREMENTS
--------------
 
The code is written in Python. Please see the requirements.txt to see what packages to install. 
To install all dependencies, 

```
pip install -r requirements.txt
```

# USAGE
-------

To run the algorithm, call `segment.py'.  It takes 6 arguments:

    1. --image: path to the image to be segmented
    2. --lamba: weight paramter for the fidelity term
	3. --beta: weight parameter for the intensity difference term
	4. --dt: time step
	5. --phase: specifies whether to perform two-phase or four-phase segmentation.
		'two': calls for two-phase segmentation
		'four': calls for four-phase segmentation
    6. --save: name of output result

For an example,

```
python3 segment.py --image './images/grayscale_two_phase/brain.png' --lamba 5 --beta 40 --dt 15 --phase two --save './output/brain_approx.png'

python3 segment.py --image './images/color_four_phase/church.jpg' --lamba 40 --beta 200 --dt 50 --phase four --save './output/church_approx.png'
```

To see more examples, please see `official_demo.sh'


================================================================================


# SOURCE CODE ORGANIZATION
--------------------------

The source code has the following files:

    - lcv.py: contains the algorithms that perform grayscale and color image segmentation (2-phase and 4-phase)
    - official_demo.ipynb: jupyter notebook that reproduces the results from Sections 6.3-6.4.
    - parameter_demo_manuscript.ipynb: jupyter notebook that does paramenter sensitivity analysis on the manuscript image
    - parameter_demo_map.ipynb: jupyter notebook that does paramenter sensitivity analysis on the topograpy image
    - parameter_demo_vessel.ipynb: jupyter notebook that does paramenter sensitivity analysis on the vessel image
    - segment.py: code to call on the command terminal to segment image
    - utils.py: contains functions to perform miscellaneous image processing operations
    - compute_energy.py: contains functions to compute energy functions
    - official_demo.sh: contains several examples how to call segment.py
    - requirements.txt: lists the package dependencies

The source code has the following folders:
    - images: contains all the demo images
    - output: where all the output results for the paper are stored


================================================================================

# THANKS
--------

The authors would be grateful to receive any comment, especially about
portability issues, errors, bugs or strange results.
