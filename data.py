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

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
transform0 = A.Compose([
    # A.Resize(width=256, height=256),
    A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    # ToTensorV2()
])

transform = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.25),
    A.VerticalFlip(p=0.25),
    A.Rotate(p=0.25),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2()
])


def get_train_loader(root_dir='training_data'):
    l = os.listdir(root_dir)
    data = []
    opt = Option()
    n_total_steps = len(l)
    for i in range(0, len(l)):
        img = Image.open(os.path.join(root_dir + '/' + l[i] + '/' + "image.png")).convert("RGB")
        mask1 = Image.open(os.path.join(root_dir + '/' + l[i] + '/' + "mask.png")).convert('L')
        mask1 = np.asarray(mask1)
        # print(mask1.shape)
        # plt.imshow(mask1)
        # plt.show()
        k = np.expand_dims(mask1, axis=-1)
        # plt.imshow(k)
        # plt.show()
        # print("Size of k: ", k.shape)
        img = np.array(img, dtype=np.uint8)
        mask = np.array(k, dtype=np.uint8)
        transformed = transform0(image=img, mask=mask)
        p = transformed['image']
        q = transformed['mask']
        p = torch.from_numpy(p)
        q = torch.from_numpy(q)
        # p = torch.from_numpy(p)
        # q = torch.from_numpy(q)
        p = p.type(opt.dtype)
        q = q.type(opt.dtype)
        p = torch.reshape(p, (3, IMAGE_WIDTH, IMAGE_HEIGHT))
        q = torch.reshape(q, (1, IMAGE_WIDTH, IMAGE_HEIGHT))
        q = q/255.0
        # print(q.size())
        mod_data = (p, q)
        data.append(mod_data)
        print(f'Step: [{i}/{n_total_steps}]')
        # if i == 200:
        #     break
    # for i in range(0, len(l)):
    #     img = cv2.imread(os.path.join(root_dir + '/' + l[i] + '/' + "image.png"))
    #     mask1 = Image.open(os.path.join(root_dir + '/' + l[i] + '/' + "mask.png")).convert('L')
    #     mask1 = np.asarray(mask1)
    #     print(mask1.shape)
    #     # plt.imshow(mask1)
    #     # plt.show()
    #     k = np.expand_dims(mask1, axis=-1)
    #     # plt.imshow(k)
    #     # plt.show()
    #     # print("Size of k: ", k.shape)
    #     img = np.array(img, dtype=np.uint8)
    #     mask = np.array(k, dtype=np.uint8)
    #     transformed = transform(image=img, mask=mask)
    #     p = transformed['image']
    #     q = transformed['mask']
    #     # p = torch.from_numpy(p)
    #     # q = torch.from_numpy(q)
    #     p = p.type(opt.dtype)
    #     q = q.type(opt.dtype)
    #     p = torch.reshape(p, (3, 256, 256))
    #     q = torch.reshape(q, (1, 256, 256))
    #     # print(q.size())
    #     mod_data = (p, q)
    #     data.append(mod_data)
    #     print(f'Step:[{i+len(l)}/{n_total_steps}]')

    BATCH_SIZE = opt.batch_size
    dataloader_train = DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True)

    print("TRAIN DATALOADER DONE")
    return dataloader_train


def get_val_loader(root_dir='validation_data'):
    l = os.listdir(root_dir)
    data = []
    opt = Option()
    n_total_steps = len(l)
    for i in range(0, len(l)):
        img = cv2.imread(os.path.join(root_dir + '/' + l[i] + '/' + "image.png"))
        mask1 = Image.open(os.path.join(root_dir + '/' + l[i] + '/' + "mask.png")).convert('L')
        mask1 = np.asarray(mask1)
        # print(mask1.shape)
        # plt.imshow(mask1)
        # plt.show()
        k = np.expand_dims(mask1, axis=-1)
        # plt.imshow(k)
        # plt.show()
        # print("Size of k: ", k.shape)
        img = np.array(img, dtype=np.uint8)
        mask = np.array(k, dtype=np.uint8)
        transformed = transform0(image=img, mask=mask)
        p = transformed['image']
        q = transformed['mask']
        p = torch.from_numpy(p)
        q = torch.from_numpy(q)
        # p = torch.from_numpy(p)
        # q = torch.from_numpy(q)
        p = p.type(opt.dtype)
        q = q.type(opt.dtype)
        q = q/255.0
        # p = torch.reshape(p, (3, 256, 256))
        # q = torch.reshape(q, (1, 256, 256))
        p = torch.reshape(p, (3, IMAGE_WIDTH, IMAGE_HEIGHT))
        q = torch.reshape(q, (1, IMAGE_WIDTH, IMAGE_HEIGHT))
        # print(p.size())
        mod_data = (p, q)
        data.append(mod_data)
        print(f'Step:[{i}/{n_total_steps}]')
        # if i == 50:
        #     break
    BATCH_SIZE = opt.batch_size
    dataloader_train = DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True)

    print("VAL DATALOADER DONE")
    return dataloader_train


def get_test_loader(root_dir='testing_data'):
    l = os.listdir(root_dir)
    data = []
    opt = Option()
    n_total_steps = len(l)
    for i in range(0, len(l)):
        img = cv2.imread(os.path.join(root_dir + '/' + l[i] + '/' + "image.png"))
        mask1 = Image.open(os.path.join(root_dir + '/' + l[i] + '/' + "mask.png")).convert('L')
        mask1 = np.asarray(mask1)
        print(mask1.shape)
        # plt.imshow(mask1)
        # plt.show()
        k = np.expand_dims(mask1, axis=-1)
        # plt.imshow(k)
        # plt.show()
        # print("Size of k: ", k.shape)
        img = np.array(img, dtype=np.uint8)
        mask = np.array(k, dtype=np.uint8)
        transformed = transform0(image=img, mask=mask)
        p = transformed['image']
        q = transformed['mask']
        # p = torch.from_numpy(p)
        # q = torch.from_numpy(q)
        p = p.type(opt.dtype)
        q = q.type(opt.dtype)
        p = torch.reshape(p, (3, 256, 256))
        q = torch.reshape(q, (1, 256, 256))
        # print(q.size())
        mod_data = (p, q)
        data.append(mod_data)
        print(f'Step:[{i}/{n_total_steps}]')
    BATCH_SIZE = opt.batch_size
    dataloader_train = DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


if __name__ == '__main__':
    x = get_train_loader()
    print("Trainloader okay")
    # y = get_val_loader()
    # print("Valloader okay")
    # z = get_test_loader()
    # print("Testloader okay")

# class CustomDataset(Dataset):
#     def __init__(self):
#         self.imgs_path = "training_data/"
#         self.img_dim = (256,256)
#         file_list = os.listdir(self.imgs_path)
#         # print(file_list)
#         self.data = []

#         count = 0
#         for folder in file_list:
#             img_path =
#         # print("Example of data is:")
#         # print(self.data[0])
#         random.shuffle(self.data)
#         print("Number of classes: ",count)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path, class_name = self.data[idx]
#         #print(img_path)
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, self.img_dim)
#         class_id = self.class_map[class_name]
#         img_tensor = torch.from_numpy(img)
#         #img_tensor = img_tensor.permute(2, 0, 1)
#         class_id = torch.tensor([class_id])
#         return img_tensor, class_id
