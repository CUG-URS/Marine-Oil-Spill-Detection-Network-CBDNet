import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time
from networks.CBDNet import CBDNet

from framework import MyFrame
#from loss.dice_loss import SSLoss
from dice_bce_loss import dice_bce_loss
from boundary_bce_loss2 import jointloss
#from boundary_loss import BoundaryLoss
from data import ImageFolder
from hyjoint_loss import hyjoint_loss
SHAPE = (256,256)
ROOT = '.datasets/train/palsar/'
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
trainlist = list(map(lambda x: x[:-8], imagelist))
NAME = 'palsar_CBDNet'
BATCHSIZE_PER_CARD = 4

solver = MyFrame(CBDNet, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    #shuffle=True,
    shuffle=False,
    num_workers=4)

mylog = open('logs/' + NAME + '.log', 'w')
tic = time()
no_optim = 0
total_epoch = 80
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)

    print('--------')
    print(f'epoch:{epoch}, time：{int(time()-tic)}， train_loss{train_epoch_loss}')
    # print >> mylog, 'epoch:',epoch, '    time:',int(time()-tic), '    train_loss:',train_epoch_loss
    # print
    # 'epoch:', epoch, '    time:', int(time() - tic)
    # print
    # 'train_loss:', train_epoch_loss
    # print
    # 'SHAPE:', SHAPE

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + '.th')
    if no_optim > 6:
        print (mylog, 'early stop at %d epoch' % epoch)
        # print
        # 'early stop at %d epoch' % epoch
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/' + NAME + '.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

print( mylog, 'Finish!') 
# print
# 'Finish!'
mylog.close()