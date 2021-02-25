import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import cv2
import glob
from PIL import Image 
from tqdm import tqdm
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

from utils.utils import set_paths, make_abcd_dataset, MyDataset, MyDataLoader, PairDataLoader
from utils.trainer import step2_train, step2_val
from models.models import build_CNN, build_Classifier
from utils.visualizer import get_feat, show_UMAP_2D

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type=str, default='AlexNet', help='AlexNet or ResNet')
parser.add_argument('-o', '--output', type=str, default='logs/test_step2_theta1/')
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-g', '--gpuid', type=int, default=0)

args = parser.parse_args()

##########################################################################
# Config
##########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

num_epochs = args.epoch
output_dir = os.path.join(os.getcwd(), args.output)
method=args.method
batch_size=16
theta = 1
step0_path = 'logs/step0_pretrained_ac'
step1_path = 'logs/step1_pretrained_ac'

if method=='AlexNet':
    classifier2 = build_Classifier(method, cls_num=10)
    print('useing AlexNet')
elif method=='ResNet':
    classifier2 = build_Classifier(method, cls_num=10)
    print('using ResNet')
else:
    assert False, 'method Error: check -m(-method) option'
    
s2 = build_CNN(method)
t = build_CNN(method)

s2.load_state_dict(torch.load(os.path.join(step1_path, 'step1_s1.pth')))
t.load_state_dict(torch.load(os.path.join(step1_path, 'step1_s1.pth')))
# classifier2.load_state_dict(torch.load(os.path.join(step0_path, 'step0_classifier.pth')))
for param in t.parameters():
    param.requires_grad = False

l2_criterion = nn.MSELoss()
softmax_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([{'params': s2.parameters()},
                        {'params': classifier2.parameters()}], lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s2.to(device)
t.to(device)
classifier2.to(device)
l2_criterion.to(device)
softmax_criterion.to(device)

##########################################################################
# Data
##########################################################################

root = '/mnt/aoni04/jsakuma/data'
mnist_train = set_paths(root, 'mnist', 'train')
mnist_test = set_paths(root, 'mnist', 'test')
mnist_m_train = set_paths(root, 'mnist-m', 'train')
mnist_m_test = set_paths(root, 'mnist-m', 'test')

(X_a_train, y_a_train), (X_b_train, y_b_train), (X_c_train, y_c_train),(X_d_train, y_d_train) = make_abcd_dataset(mnist_train, mnist_m_train, max_num=5000, cls_flg=False)
(X_a_test, y_a_test), (X_b_test, y_b_test), (X_c_test, y_c_test), (X_d_test, y_d_test) = make_abcd_dataset(mnist_test, mnist_m_test, max_num=800, cls_flg=False)

X_ac_train = np.concatenate([X_a_train, X_c_train])
y_ac_train = np.concatenate([y_a_train, y_c_train])
d_ac_train = np.concatenate([np.zeros(len(y_a_train)), np.ones(len(y_c_train))])
X_ac_test = np.concatenate([X_a_test, X_c_test])
y_ac_test = np.concatenate([y_a_test, y_c_test])
d_ac_test = np.concatenate([np.zeros(len(y_a_test)), np.ones(len(y_c_test))])

d_a_train =  np.zeros(len(y_a_train))
d_a_test = np.zeros(len(y_a_test))
d_c_train = np.ones(len(y_c_train))
d_c_test = np.ones(len(y_c_test))

d_b_train =  np.zeros(len(y_b_train))
d_b_test = np.zeros(len(y_b_test))
d_d_train = np.ones(len(y_d_train))
d_d_test = np.ones(len(y_d_test))

# transform
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#データセットの作成
batch_size=16
ds_ac_train = MyDataset(X_ac_train, y_ac_train, d_ac_train, transform_train)
ds_a_train = MyDataset(X_a_train, y_a_train, d_a_train, transform_train)
ds_b_train = MyDataset(X_b_train, y_b_train, d_b_train, transform_train)
ds_c_train = MyDataset(X_c_train, y_c_train, d_c_train, transform_train)
ds_d_train = MyDataset(X_d_train, y_d_train, d_d_train, transform_train)

ds_ac_test = MyDataset(X_ac_test, y_ac_test, d_ac_test, transform_test)
ds_a_test = MyDataset(X_a_test, y_a_test, d_a_test, transform_test)
ds_b_test = MyDataset(X_b_test, y_b_test, d_b_test, transform_test)
ds_c_test = MyDataset(X_c_test, y_c_test, d_c_test, transform_test)
ds_d_test = MyDataset(X_d_test, y_d_test, d_d_test, transform_test)

#loaderの作成
loader_ac_train = MyDataLoader(ds_ac_train, batch_size=batch_size, shuffle=True)
loader_a_train = MyDataLoader(ds_a_train, batch_size=batch_size, shuffle=True)
loader_b_train = MyDataLoader(ds_b_train, batch_size=batch_size, shuffle=True)
loader_c_train = MyDataLoader(ds_c_train, batch_size=batch_size, shuffle=True)
loader_d_train = MyDataLoader(ds_d_train, batch_size=batch_size, shuffle=True)

loader_ac_test = MyDataLoader(ds_ac_test, batch_size=batch_size, shuffle=False)
loader_a_test = MyDataLoader(ds_a_test, batch_size=batch_size, shuffle=False)
loader_b_test = MyDataLoader(ds_b_test, batch_size=batch_size, shuffle=False)
loader_c_test = MyDataLoader(ds_c_test, batch_size=batch_size, shuffle=False)
loader_d_test = MyDataLoader(ds_d_test, batch_size=batch_size, shuffle=False)

loader_pair_ac_train = PairDataLoader(ds_a_train, ds_c_train, batch_size=batch_size, shuffle=True)
loader_pair_bd_train = PairDataLoader(ds_b_train, ds_d_train, batch_size=batch_size, shuffle=True)

loader_pair_ac_test = PairDataLoader(ds_a_test, ds_c_test, batch_size=1, shuffle=False)
loader_pair_bd_test = PairDataLoader(ds_b_test, ds_d_test, batch_size=1, shuffle=False)

##########################################################################
# Train/Val Model
##########################################################################
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/weights', exist_ok=True)
os.makedirs(output_dir+'/figures', exist_ok=True)
os.makedirs(output_dir+'/history', exist_ok=True)
best_acc = 0
train_steps = len(loader_pair_ac_train)
val_steps = len(loader_pair_ac_test)
Acc = {'train': [], 'val': []}
Loss_cls = {'train': [], 'val': []}
Loss_l2 = {'train': [], 'val': []}
RESULT = {'Acc_A': [], 'Acc_B': [], 'Acc_C': [], 'Acc_D': [], 'Lcls_A': [], 'Lcls_B': [], 'Lcls_C': [], 'Lcls_D': [],
          'L2_AC': [], 'L2_BD': []}


val_lossAC, val_accAC, acc_sA, acc_tC, loss_sA, loss_tC, l2_lossAC = step2_val(val_steps, s2, t, classifier2, loader_pair_ac_test, softmax_criterion, l2_criterion, device)
            
val_lossBD, val_accBD, acc_sB, acc_tD, loss_sB, loss_tD, l2_lossBD = step2_val(val_steps, s2, t, classifier2, loader_pair_bd_test, softmax_criterion, l2_criterion, device)

RESULT['Acc_A'].append(acc_sA)
RESULT['Acc_B'].append(acc_sB)
RESULT['Acc_C'].append(acc_tC)
RESULT['Acc_D'].append(acc_tD)
RESULT['Lcls_A'].append(loss_sA)
RESULT['Lcls_B'].append(loss_sB)
RESULT['Lcls_C'].append(loss_tC)
RESULT['Lcls_D'].append(loss_tD)
RESULT['L2_AC'].append(l2_lossAC)
RESULT['L2_BD'].append(l2_lossBD)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')

    for phase in ['train', 'val']:
        # train
        if phase == 'train':
            train_loss, train_acc, l2_loss, softmax_loss = step2_train(train_steps, s2, t, classifier2, loader_b_train, 
                                                                      loader_pair_ac_train, l2_criterion, softmax_criterion,
                                                                      device, theta, optimizer)
            # logging
            Acc[phase].append(train_acc)
            Loss_cls[phase].append(softmax_loss)
            Loss_l2[phase].append(l2_loss)
            
        # val
        else:
            val_lossAC, val_accAC, acc_sA, acc_tC, loss_sA, loss_tC, l2_lossAC = step2_val(val_steps, s2, t, classifier2, loader_pair_ac_test, softmax_criterion, l2_criterion, device)
            
            val_lossBD, val_accBD, acc_sB, acc_tD, loss_sB, loss_tD, l2_lossBD = step2_val(val_steps, s2, t, classifier2, loader_pair_bd_test, softmax_criterion, l2_criterion, device)
            
            # save model
            torch.save(s2.state_dict(), os.path.join(output_dir, 'weights/step2_s2_epoch{}_acc{:.3f}.pth'.format(epoch, acc_tD)))
            torch.save(t.state_dict(), os.path.join(output_dir, 'weights/step2_t_epoch{}_acc{:.3f}.pth'.format(epoch, acc_tD)))
            torch.save(classifier2.state_dict(), os.path.join(output_dir, 'weights/step2_classifier_epoch{}_acc{:.3f}.pth'.format(epoch, acc_tD)))
            
            # logging
            Acc[phase].append(val_accAC)
            Loss_cls[phase].append(loss_sB)
            Loss_l2[phase].append(l2_lossAC)
            
            RESULT['Acc_A'].append(acc_sA)
            RESULT['Acc_B'].append(acc_sB)
            RESULT['Acc_C'].append(acc_tC)
            RESULT['Acc_D'].append(acc_tD)
            RESULT['Lcls_A'].append(loss_sA)
            RESULT['Lcls_B'].append(loss_sB)
            RESULT['Lcls_C'].append(loss_tC)
            RESULT['Lcls_D'].append(loss_tD)
            RESULT['L2_AC'].append(l2_lossAC)
            RESULT['L2_BD'].append(l2_lossBD)
            
            
    # plot feature by UMAP
    target_dictAC = {'0':0, '1': 0, '2': 0, '3': 0, '4': 0} 
    target_dictBD = {'5':0, '6': 0, '7': 0, '8': 0, '9': 0} 
    featA, featC = get_feat(s2, t, loader_pair_ac_test, device, target_dictAC, N=10)
    featB, featD = get_feat(s2, t, loader_pair_bd_test, device, target_dictBD, N=10)

    featAB = featA+featB
    featABCD = featA+featB+featC+featD
    show_UMAP_2D(featAB, featABCD, os.path.join(output_dir, 'figures/umap_2d_plot_epoch{}.png'.format(epoch)))

        
##########################################################################
# Save Log
##########################################################################
df = pd.DataFrame(RESULT)
df.to_csv(os.path.join(output_dir, 'result.csv'), encoding='utf_8_sig')

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
plt.plot(Acc['val'], label='val')
plt.plot(Acc['train'], label='train')
plt.legend()
plt.savefig(os.path.join(output_dir, 'history/history_acc.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
plt.plot(RESULT['Acc_A'], label='A')
plt.plot(RESULT['Acc_B'], label='B')
plt.plot(RESULT['Acc_C'], label='C')
plt.plot(RESULT['Acc_D'], label='D')
plt.legend()
plt.savefig(os.path.join(output_dir, 'history/history_acc_ABCD.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
plt.plot(Loss_cls['train'], label='train')
plt.plot(Loss_cls['val'], label='val')
plt.legend()
plt.savefig(os.path.join(output_dir, 'history/history_cls_loss.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
plt.plot(Loss_l2['train'], label='train')
plt.plot(Loss_l2['val'], label='val')
plt.legend()
plt.savefig(os.path.join(output_dir, 'history/history_l2_loss.png'))
plt.close()