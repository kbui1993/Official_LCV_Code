# two-phase grayscale segmentation: brain
python3 segment.py --image './images/grayscale_two_phase/brain.png' --lamba 5 --beta 40 --dt 15 --phase two --save './output/brain_approx.png'

# four-phase grayscale segmentation: microscopy image 1
python3 segment.py --image './images/grayscale_four_phase/microscopy1.png' --lamba 10 --beta 50 --dt 40 --phase four --save './output/microscopy_approx.png'

# four-phase grayscale segmentation: microscopy image 2
python3 segment.py --image './images/grayscale_four_phase/microscopy2.png' --lamba 20 --beta 80 --dt 30 --phase four --save './output/microscopy2_approx.png'

# two-phase color segmentation: bird
python3 segment.py --image './images/color_two_phase/bird2.png' --lamba 30 --beta 100 --dt 10 --phase two --save './output/bird_approx.png'

# two-phase color segmentation: helicopter
python3 segment.py --image './images/color_two_phase/chopper.png' --lamba 50 --beta 50 --dt 20 --phase two --save './output/chopper_approx.png'

# four-phase color segmentation: church
python3 segment.py --image './images/color_four_phase/church.jpg' --lamba 40 --beta 200 --dt 50 --phase four --save './output/church_approx.png'

# four-phase color segmentation: butterfly
python3 segment.py --image './images/color_four_phase/butterfly.jpg' --lamba 40 --beta 60 --dt 60 --phase four --save './output/butterfly_approx.png'

