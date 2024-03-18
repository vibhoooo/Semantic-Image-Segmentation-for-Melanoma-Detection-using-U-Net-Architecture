import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from dataset import get_train_valid_loader, get_test_loader
from math_UNet import UNet as MODEL
# from model import UNet2 as MODEL
from utils import Option, encode_and_save, compute_iou
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from data import get_train_loader, get_val_loader


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT > threshold

    # TP : True Positive
    # FN : False Negative
    TP = torch.logical_and(SR, GT)
    FN = torch.logical_and((SR == 0), (GT == 1))

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold

    # TN : True Negative
    # FP : False Positive
    TN = torch.logical_and((SR == 0), (GT == 0))
    FP = torch.logical_and((SR == 1), (GT == 0))

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold

    # TP : True Positive
    # FP : False Positive
    TP = torch.logical_and(SR, GT)
    P = SR

    PC = float(torch.sum(TP)) / (float(torch.sum(P)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT > threshold
    Inter = torch.sum(torch.logical_and(SR, GT))
    Union = torch.sum(torch.logical_or(SR, GT))

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT > threshold
    Inter = torch.sum(torch.logical_and(SR, GT)).item()
    Union = torch.sum(torch.logical_or(SR, GT)).item()
    DC = float(2 * Inter) / (float(Inter + Union) + 1e-6)

    return DC


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = Option()
    model = MODEL(input_channels=3, nclasses=1)
    model.to(device)
    train_loader = get_train_loader()
    # val_loader = get_val_loader()
    # train_loader = get_val_loader()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 0.00001)
    criterion = nn.BCEWithLogitsLoss().cuda()
    los = 0
    num_batches = 0
    accuracy_train = 0
    sensitivity_train = 0
    specificity_train = 0
    precision_train = 0
    f1_train = 0
    js_train = 0
    dc_train = 0

    avg_loss = []
    avg_accuracy_train = []
    avg_sensitivity_train = []
    avg_specificity_train = []
    avg_precision_train = []
    avg_f1_train = []
    avg_js_train = []
    avg_dc_train = []
    epoch_store = []

    n_total_steps = len(train_loader)
    for epoch in range(0, opt.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # data = sample_batched['image']
            # target = sample_batched['mask']
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
            # print(f'Data max is: {data.max()}')
            # print(f'Target max is: {target.max()}')
            optimizer.zero_grad()
            output = model(data)
            output = torch.sigmoid(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{opt.epochs}], Step [{batch_idx + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            los += loss.item()
            accuracy_train += get_accuracy(output, target)
            sensitivity_train += get_sensitivity(output, target)
            specificity_train += get_specificity(output, target)
            precision_train += get_precision(output, target)
            f1_train += get_F1(output, target)
            js_train += get_JS(output, target)
            dc_train += get_DC(output, target)
            # break
        scheduler.step()
        avg_loss.append(los / n_total_steps)
        avg_accuracy_train.append(accuracy_train / n_total_steps)
        avg_precision_train.append(precision_train / n_total_steps)
        avg_sensitivity_train.append(sensitivity_train / n_total_steps)
        avg_specificity_train.append(specificity_train / n_total_steps)
        avg_f1_train.append(f1_train / n_total_steps)
        avg_js_train.append(js_train / n_total_steps)
        avg_dc_train.append(dc_train / n_total_steps)
        epoch_store.append(epoch + 1)
        los = 0
        accuracy_train = 0
        sensitivity_train = 0
        specificity_train = 0
        precision_train = 0
        f1_train = 0
        js_train = 0
        dc_train = 0

    datas = {'Average Training Accuracy': avg_accuracy_train,
             'Average Training Loss': avg_loss,
             'Average Training Precision': avg_precision_train,
             'Average Training Sensitivity': avg_precision_train,
             'Average Training Specificity': avg_specificity_train,
             'Average Training F1': avg_f1_train,
             'Average Training JS': avg_js_train,
             'Average Training Dice Coefficient': avg_dc_train,
             'Epochs': epoch_store
             }

    df = pd.DataFrame(datas)
    df.to_csv("UNET_MODEL_METRICS_math.csv")
    if opt.save_model:
        torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, 'UNET_Model_math.pt'))

    print('TRAIN 1 program run complete')