import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


##########################################################################
# Step 0
##########################################################################
def step0_train(num_steps, cnn, classifier,
                dataloader, softmax_criterion,
                device, optimizer):
    
    dataloader.on_epoch_end()
    cnn.train()
    classifier.train()
#     scheduler.step()
#     print('lr:{}'.format(scheduler.get_lr()[0]))

    epoch_loss = 0.0
    train_cnt = 0
    correct = 0.
    for idx, batch in zip(tqdm(range(num_steps)), dataloader):
        X, y, d = batch
        X, y = X.to(device), y.to(device)        
        
        optimizer.zero_grad()
        
        # feature extraction
        feat = cnn(X)
        
        # classification
        cls = classifier(feat)
        
        loss = softmax_criterion(cls, y)
        loss.backward()
        optimizer.step()
        loss = loss.item()

        epoch_loss += loss
        loss = 0
        train_cnt += X.shape[0]
        
        cls_pred = cls.argmax(dim=1, keepdim=True)
        correct += cls_pred.eq(y.view_as(cls_pred)).sum().item()

    epoch_loss = epoch_loss / train_cnt
    cls_acc = correct / float(train_cnt)
    print('train')
    print('loss: {:.3f}'.format(epoch_loss))
    print('cls_acc: {:.3f}'.format(cls_acc))

    return epoch_loss, cls_acc


def step0_val(num_steps, cnn, classifier,
              dataloader, softmax_criterion, device):
    
    cnn.eval()
    classifier.eval()

    epoch_loss = 0.0
    train_cnt = 0
    correct = 0.
    with torch.no_grad():
        for idx, batch in zip(tqdm(range(num_steps)), dataloader):
            X, y, d = batch
            X, y = X.to(device), y.to(device)
                    
            # feature extraction
            feat = cnn(X)

            # classification
            cls = classifier(feat)

            loss = softmax_criterion(cls, y)
            loss = loss.item()

            epoch_loss += loss
            loss = 0
            train_cnt += X.shape[0]

            cls_pred = cls.argmax(dim=1, keepdim=True)
            correct += cls_pred.eq(y.view_as(cls_pred)).sum().item()

    epoch_loss = epoch_loss / train_cnt
    acc = correct / float(train_cnt)
    print('val')
    print('loss: {:.3f}'.format(epoch_loss))
    print('acc: {:.3f}'.format(acc))

    return epoch_loss, acc

##########################################################################
# Step 1
##########################################################################
def step1_train(num_steps, cnn1, cnn2, dataloader, l2_criterion, device, optimizer):
    
    dataloader.on_epoch_end()
    cnn1.train()
    cnn2.train()
#     scheduler.step()
#     print('lr:{}'.format(scheduler.get_lr()[0]))

    epoch_loss = 0.0
    train_cnt = 0.
    correct = 0.
    for idx, (batch1, batch2) in zip(tqdm(range(num_steps)), dataloader):
        X1, y1, d1 = batch1
        X1, y1 = X1.to(device), y1.to(device)
        X2, y2, d2 = batch2
        X2, y2 = X2.to(device), y2.to(device)
        
        optimizer.zero_grad()
        
        # feature extraction
        feat1 = cnn1(X1)
        feat2 = cnn2(X2)
        
        loss = l2_criterion(feat1, feat2)
        
        loss.backward()
        optimizer.step()
        loss = loss.item()

        epoch_loss += loss
        loss = 0
        train_cnt += X1.shape[0]
    

    epoch_loss = epoch_loss / train_cnt
    print('train')
    print('loss: {}'.format(epoch_loss))

    return epoch_loss


def step1_val(num_steps, cnn1, cnn2, dataloader, l2_criterion, device):
    
    cnn1.eval()
    cnn2.eval()

    epoch_loss = 0.0
    train_cnt = 0.
    with torch.no_grad():
        for idx, (batch1, batch2) in zip(tqdm(range(num_steps)), dataloader):
            X1, y1, d1 = batch1
            X1, y1 = X1.to(device), y1.to(device)
            X2, y2, d2 = batch2
            X2, y2 = X2.to(device), y2.to(device)

            # feature extraction
            feat1 = cnn1(X1)
            feat2 = cnn2(X2)

            loss = l2_criterion(feat1, feat2)

            loss = loss.item()
            epoch_loss += loss
            loss = 0
            train_cnt += X1.shape[0]

    epoch_loss = epoch_loss / train_cnt
    print('val')
    print('loss: {}'.format(epoch_loss))

    return epoch_loss


##########################################################################
# Step 2
##########################################################################
def step2_train(num_steps, cnn1, cnn2, classifier,
                dataloader_r, dataloader_ir, l2_criterion, softmax_criterion,
                device, theta, optimizer):
    
    dataloader_r.on_epoch_end()
    dataloader_ir.on_epoch_end()
    cnn1.train()
    cnn2.train()
    classifier.train()
#     scheduler.step()
#     print('lr:{}'.format(scheduler.get_lr()[0]))

    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_l2_loss = 0.0
    train_cnt = 0
    correct = 0.
    for idx, batch_r, (batch_ir1, batch_ir2) in zip(tqdm(range(num_steps)), dataloader_r, dataloader_ir):
        X_r, y_r, d_r = batch_r
        X_ir1, y_ir1, d_ir1 = batch_ir1
        X_ir2, y_ir2, d_ir2 = batch_ir2
        X_r, y_r = X_r.to(device), y_r.to(device)
        X_ir1, y_ir1 = X_ir1.to(device), y_ir1.to(device)
        X_ir2, y_ir2 = X_ir2.to(device), y_ir2.to(device)        
        
        optimizer.zero_grad()
        
        # feature extraction
        feat_r = cnn1(X_r)
        feat_ir1 = cnn1(X_ir1)
        feat_ir2 = cnn2(X_ir2)
        
        # classification
        cls = classifier(feat_r)
        
        cls_loss = softmax_criterion(cls, y_r)
        l2_loss = l2_criterion(feat_ir1, feat_ir2)
        
        loss = cls_loss + l2_loss * theta
        
        loss.backward()
        optimizer.step()
        loss = loss.item()
        cls_loss = cls_loss.item()
        l2_loss = l2_loss.item()

        epoch_loss += loss
        epoch_cls_loss += cls_loss
        epoch_l2_loss += l2_loss
        loss, l2_loss, softmax_loss = 0, 0, 0
        train_cnt += X_r.shape[0]
        
        cls_pred = cls.argmax(dim=1, keepdim=True)
        correct += cls_pred.eq(y_r.view_as(cls_pred)).sum().item()

    epoch_cls_loss = epoch_cls_loss / train_cnt
    epoch_l2_loss = epoch_l2_loss / train_cnt
    epoch_loss = epoch_cls_loss + theta * epoch_l2_loss
    cls_acc = correct / float(train_cnt)
    print('train')
    print('loss: {:.3f}, cls_loss: {:.5f}, l2_loss: {:.5f}'.format(epoch_loss, epoch_cls_loss, epoch_l2_loss))
    print('cls_acc: {:.3f}'.format(cls_acc))

    return epoch_loss, cls_acc, epoch_l2_loss, epoch_cls_loss


def step2_val(num_steps, cnn1, cnn2, classifier, dataloader,
              softmax_criterion, l2_criterion, device):
    
    cnn1.eval()
    cnn2.eval()
    classifier.eval()

    epoch_loss = 0.0
    epoch_s_loss = 0.0
    epoch_t_loss = 0.0
    epoch_l2_loss = 0.0
    train_cnt = 0
    correct_s = 0.
    correct_t = 0.
    with torch.no_grad():
        for idx, (batch_r_s, batch_r_t) in zip(tqdm(range(num_steps)), dataloader):
            X_r_s, y_r_s, d_r_s = batch_r_s
            X_r_t, y_r_t, d_r_t = batch_r_t
            X_r_s, y_r_s = X_r_s.to(device), y_r_s.to(device)
            X_r_t, y_r_t = X_r_t.to(device), y_r_t.to(device)
                    
            # feature extraction
            feat_s = cnn1(X_r_s)
            feat_t = cnn2(X_r_t)

            # classification
            cls_s = classifier(feat_s)
            cls_t = classifier(feat_t)

            cls_loss_s = softmax_criterion(cls_s, y_r_s)
            cls_loss_t = softmax_criterion(cls_t, y_r_t)
            loss = cls_loss_s + cls_loss_t
            l2_loss = l2_criterion(feat_t, feat_s)

            loss = loss.item()
            s_loss = cls_loss_s.item()
            t_loss = cls_loss_t.item()
            l2_loss = l2_loss.item()

            epoch_loss += loss
            epoch_s_loss += s_loss
            epoch_t_loss += t_loss
            epoch_l2_loss += l2_loss
            loss, s_loss, t_loss = 0, 0, 0
            train_cnt += X_r_s.shape[0]

            cls_pred_s = cls_s.argmax(dim=1, keepdim=True)
            cls_pred_t = cls_t.argmax(dim=1, keepdim=True)
            correct_s += cls_pred_s.eq(y_r_s.view_as(cls_pred_s)).sum().item()
            correct_t += cls_pred_t.eq(y_r_t.view_as(cls_pred_t)).sum().item()

    epoch_s_loss = epoch_s_loss / train_cnt
    epoch_t_loss = epoch_t_loss / train_cnt
    epoch_l2_loss = epoch_l2_loss / train_cnt
    epoch_loss = epoch_s_loss + epoch_t_loss
    acc_s = correct_s / float(train_cnt)
    acc_t = correct_t / float(train_cnt)
    acc = float((acc_s + acc_t) / 2)
    print('val')
    print('loss: {:.3f}, source loss: {:.5f}, target loss: {:.5f}, l2 loss: {:.5f}'.format(epoch_loss, epoch_s_loss, epoch_t_loss, epoch_l2_loss))
    print('acc: {:.3f}, source acc: {:.3f}, target acc: {:.3f}'.format(acc, acc_s, acc_t))

    return epoch_loss, acc, acc_s, acc_t, epoch_s_loss, epoch_t_loss, epoch_l2_loss