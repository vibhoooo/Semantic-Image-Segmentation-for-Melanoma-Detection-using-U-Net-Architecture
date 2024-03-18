import sys

import matplotlib.pyplot as plt
import cv2
from model import UNet2

model = UNet2(input_channels=3, nclasses=1)

path = sys.argv[1]
image = cv2.imread()
