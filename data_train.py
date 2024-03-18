import os
import shutil
from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# from skimage import io, transform
from utils import Option
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def get_train_loader(root_dir='training_data'):
    l = os.listdir(root_dir)
    data = []
    opt = Option()
    n_total_steps = len(l)
    for i in range(0, len(l)):
        mask1 = Image.open(os.path.join(root_dir + '/' + l[i] + '/' + "mask.png")).convert('L')
        mask1 = np.asarray(mask1)
        k = np.expand_dims(mask1, axis=-1)
        data = Image.fromarray(k)
        data.save("fuck.png")
        break



if __name__ == "__main__":
    get_train_loader()
    print("TRAINLOADER DONE")
