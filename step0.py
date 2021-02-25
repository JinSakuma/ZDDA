import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import glob
from PIL import Image 
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

from utils.utils import set_paths, make_abcd_dataset, MyDataset, MyDataLoader, PairDataLoader
from utils.trainer import step0_train, step0_val
from models.models import build_CNN, build_Classifier


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type=str, default='AlexNet', help='AlexNet or ResNet')
parser.add_argument('-o', '--output', type=str, default='logs/5cls/step0_2/')
parser.add_argument('-e', '--epoch', type=int, default=20)
parser.add_argument('-g', '--gpuid', type=int, default=0)

args = parser.parse_args()

##########################################################################
# Config
##########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

num_epochs = args.epoch
output_dir = args.output
method=args.method
batch_size=16
theta = 1.0

if method=='AlexNet':
    classifier0 = build_Classifier(method, cls_num=5)
    print('useing AlexNet')
elif method=='ResNet':
    classifier0 = build_Classifier(method, cls_num=5)
    print('using ResNet')
else:
    assert False, 'method Error: check -m(-method) option'
    
t = build_CNN(method)


softmax_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([{'params': t.parameters()},
                        {'params': classifier0.parameters()}], lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t.to(device)
classifier0.to(device)
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

# X_ab_train = np.concatenate([X_a_train, X_b_train])
# y_ab_train = np.concatenate([y_a_train, y_b_train])
# d_ab_train = np.concatenate([np.zeros(len(y_a_train)), np.ones(len(y_b_train))])
X_ac_train = np.concatenate([X_a_train, X_c_train])
y_ac_train = np.concatenate([y_a_train, y_c_train])
d_ac_train = np.concatenate([np.zeros(len(y_a_train)), np.ones(len(y_c_train))])
# X_ab_test = np.concatenate([X_a_test, X_b_test])
# y_ab_test = np.concatenate([y_a_test, y_b_test])
# d_ab_test = np.concatenate([np.zeros(len(y_a_test)), np.ones(len(y_b_test))])
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
# ds_ab_train = MyDataset(X_ab_train, y_ab_train, d_ab_train, transform_train)
ds_ac_train = MyDataset(X_ac_train, y_ac_train, d_ac_train, transform_train)
ds_a_train = MyDataset(X_a_train, y_a_train, d_a_train, transform_train)
ds_b_train = MyDataset(X_b_train, y_b_train, d_b_train, transform_train)
ds_c_train = MyDataset(X_c_train, y_c_train, d_c_train, transform_train)
ds_d_train = MyDataset(X_d_train, y_d_train, d_d_train, transform_train)

# ds_ab_test = MyDataset(X_ab_test, y_ab_test, d_ab_test, transform_test)
ds_ac_test = MyDataset(X_ac_test, y_ac_test, d_ac_test, transform_test)
ds_a_test = MyDataset(X_a_test, y_a_test, d_a_test, transform_test)
ds_b_test = MyDataset(X_b_test, y_b_test, d_b_test, transform_test)
ds_c_test = MyDataset(X_c_test, y_c_test, d_c_test, transform_test)
ds_d_test = MyDataset(X_d_test, y_d_test, d_d_test, transform_test)

#loaderの作成
# loader_ab_train = MyDataLoader(ds_ab_train, batch_size=batch_size, shuffle=True)
loader_ac_train = MyDataLoader(ds_ac_train, batch_size=batch_size, shuffle=True)
loader_a_train = MyDataLoader(ds_a_train, batch_size=batch_size, shuffle=True)
loader_b_train = MyDataLoader(ds_b_train, batch_size=batch_size, shuffle=True)
loader_c_train = MyDataLoader(ds_c_train, batch_size=batch_size, shuffle=True)
loader_d_train = MyDataLoader(ds_d_train, batch_size=batch_size, shuffle=True)

# loader_ab_test = MyDataLoader(ds_ab_test, batch_size=batch_size, shuffle=False)
loader_ac_test = MyDataLoader(ds_ac_test, batch_size=batch_size, shuffle=False)
loader_a_test = MyDataLoader(ds_a_test, batch_size=batch_size, shuffle=False)
loader_b_test = MyDataLoader(ds_b_test, batch_size=batch_size, shuffle=False)
loader_c_test = MyDataLoader(ds_c_test, batch_size=batch_size, shuffle=False)
loader_d_test = MyDataLoader(ds_d_test, batch_size=batch_size, shuffle=False)

# loader_pair_ac_train = PairDataLoader(ds_a_train, ds_c_train, batch_size=batch_size, shuffle=True)
# loader_pair_bd_train = PairDataLoader(ds_b_train, ds_d_train, batch_size=batch_size, shuffle=True)

# loader_pair_ac_test = PairDataLoader(ds_a_test, ds_c_test, batch_size=batch_size, shuffle=False)
# loader_pair_bd_test = PairDataLoader(ds_b_test, ds_d_test, batch_size=batch_size, shuffle=False)

##########################################################################
# Step 2
##########################################################################
os.makedirs(output_dir, exist_ok=True)
best_acc = 0
train_steps = len(loader_ac_train)
val_steps = 500
Loss = {'train': [], 'val': []}
Acc = {'train': [], 'val': []}
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')

    for phase in ['train', 'val']:
        if phase == 'train':
            train_loss, train_acc = step0_train(train_steps, t, classifier0,
                                                loader_ac_train, softmax_criterion,
                                                device, optimizer)
            Loss[phase].append(train_loss)
            Acc[phase].append(train_acc)
        else:
            val_loss, val_acc = step0_val(val_steps, t, classifier0,
                                          loader_ac_test, softmax_criterion, device)
            Loss[phase].append(val_loss)
            Acc[phase].append(val_acc)
            if best_acc < val_acc:
                torch.save(t.state_dict(), os.path.join(output_dir, 'step0_t.pth'))
                torch.save(classifier0.state_dict(), os.path.join(output_dir, 'step0_classifier.pth'))
                best_acc = val_acc
              
        
plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
plt.plot(Loss['val'], label='val')
plt.plot(Loss['train'], label='train')
plt.legend()
plt.savefig(os.path.join(output_dir, 'history_loss.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
plt.plot(Acc['val'], label='val')
plt.plot(Acc['train'], label='train')
plt.legend()
plt.savefig(os.path.join(output_dir, 'history_acc.png'))
plt.close()