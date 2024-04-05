# Import raw EEG data 
import mne_bids as mb
import mne 
import torch
import pytorch_lightning as pl
import numpy as np 

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


import data_utils4 as du
from LinearModel import linearModel

def mytransform(raw):
    raw.filter(0.1,40)
    raw._data=raw._data*1e6
    return raw

pl.seed_everything(42, workers=True)

batchSize= 10000
channelIdxs=[1,19,23] 
valSub=0
beforePts=500
afterPts=500
targetPts=100

bidsPath= 'Y:\\NTdata\\BIDS\\EESM17\\'
subjectIds=mb.get_entity_vals(bidsPath,'subject', with_key=False)
trainIds=subjectIds.copy()
trainIds.pop(valSub)

for _ in range(6): # Remove some files 
    trainIds.pop(0)

# Test import 
# temp=mb.BIDSPath(root=bidsPath,subject='001',session='001',task='sleep',datatype='eeg',extension='.set',check=False)
# tempRaw=mne.io.read_raw_eeglab(str(temp), preload=True,verbose=False)
# tempRaw.plot()

# channelsToExclude=(1- np.isin(range(0,tempRaw.info['nchan']),channelIdxs)).nonzero()[0].astype('int')
# channelsToExclude=np.asarray(tempRaw.ch_names)[channelsToExclude]
# tempRaw.drop_channels(channelsToExclude)
# tempRaw.plot()

# tempRaw.filter(0.1, 40)
# tempRaw.plot()

# tempRaw._data=tempRaw._data*1e6
# tempRaw.plot()

# if mytransform:
#     tempRaw = mytransform(tempRaw)
    
# tempRaw.plot()

# setFilePaths = mb.find_matching_paths(root=ROOT,extensions='.set',datatypes='eeg')
# file = setFilePaths[0]
# raw=mne.io.read_raw_eeglab(file, preload=True)
# raw.plot()

trainPaths=du.returnFilePaths(bidsPath,trainIds,sessionIds=['001']) # There is onlyone session in small dataset
valPaths=du.returnFilePaths(bidsPath,[subjectIds[valSub]],sessionIds=['001'])

print('Loading training data')
ds_train=du.EEG_dataset_from_paths(trainPaths, beforePts=beforePts,
                                   afterPts=afterPts,targetPts=targetPts, 
                                   channelIdxs=channelIdxs,preprocess=False,
                                    limit=None,train_size = 100000,
                                   transform=mytransform)
dl_train=torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True)

print('Loading validation data, subject = ' + subjectIds[valSub])
ds_val=du.EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,
                                 targetPts=targetPts, channelIdxs=1,
                                 preprocess=False,limit=100000,
                                 transform=mytransform)
dl_val=torch.utils.data.DataLoader(ds_val, batch_size=batchSize)

######################### Looking into data structure #########################
# Look at dataset function
# import numpy as np
# import mne 

# raws = []
# bidsPaths = trainPaths.copy()
# file_paths = [str(fp) for fp in bidsPaths]
# nfilesToLoad = len(file_paths)
# fileIdxToLoad=np.random.choice(len(file_paths),nfilesToLoad,replace=False)
# fileIdx = fileIdxToLoad[0]
# tempRaw=mne.io.read_raw_eeglab(file_paths[fileIdx],preload=True,verbose=False)

# channelsToExclude=(1- np.isin(range(0,tempRaw.info['nchan']),channelIdxs)).nonzero()[0].astype('int')
# channelsToExclude=np.asarray(tempRaw.ch_names)[channelsToExclude]
# tempRaw.drop_channels(channelsToExclude)
# raws.append(tempRaw)

# Look at dataloader
# All are double lists
# x, y = next(iter(dl_train))

# x1, x2 = x
# Xinput=torch.cat((x1,x2),dim=1) 

# from LinearModel import linearRegression
# lin_reg = linearRegression(beforePts+afterPts, targetPts) 
# pred = lin_reg(Xinput)

# batch = next(iter(dl_train))
# lin_model = linearModel(0.01, beforePts+afterPts, targetPts, 20)
# lin_model.forward(x)
# lin_model.training_step(batch, 0)

# pred = lin_model.forward(Xinput)
# abs(pred - y)
# metric = torch.nn.MSELoss()
# loss = metric(pred, y)

############################# Testing Linear Model ############################
# Define model 
# Why is there 100.000 trainable parameters like the limit and not 1000 like the size of each sample? 
lin_model = linearModel(0.001, 1000, 100, warmup=1, 
                        max_iters=10) 
# metric = torch.nn.MSELoss()

# # Fit model 
trainer = pl.Trainer(devices="auto",accelerator="auto",
                      max_epochs=2)

trainer.fit(lin_model, dl_train, dl_val)

########################## Testing Batch Norm ##########################
context_block = 50
num_heads=1 
input_dim = 50 
dim_ff = 2*input_dim
patch_length = int(1000/context_block)
dropout = 0.2

train_iter = iter(dl_train)
x, y = next(train_iter)
x1, x2 = x   # x1 is 10000 x 500 [batch_size x context_size]
x1T = x1.reshape(x1.shape[0],context_block,-1) # 10000 x 50 x 10 [ batch_size x context_block x context_size/context_block]
x1T = torch.transpose(x1T,1,2) # 10000 x 10 x 50 [batch_size x context_size/context_block x context_block]

x2T = x2.reshape(x2.shape[0],context_block,-1)
x2T = torch.transpose(x2T,1, 2)
x =torch.cat((x1T,x2T),dim=1) # 10000 x 20 x 50 [batch_size, no patches, context_block]

import torch.nn as nn
self_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                           num_heads=num_heads,
                                           batch_first=True)

# Two-layer MLP
linear_net = nn.Sequential(nn.Linear(input_dim, dim_ff),
                                nn.ReLU(),
                                nn.Linear(dim_ff, input_dim)
                                )

# Layers to apply in between the main layers
batchNorm1 = nn.BatchNorm1d(input_dim) # Input is number of "channels"
batchNorm2 = nn.BatchNorm1d(input_dim)
dropout = nn.Dropout(dropout)
    

attn_out = self_attn(x, x, x)[0] # output is tuple therefore [0] | out size [batch_size, 20, input_dim]
x = x + dropout(attn_out) # out size as above
x = torch.transpose(x,1, 2)
x = batchNorm1(x)
x = torch.transpose(x,1, 2)

# MLP part
linear_out = linear_net(x)
x = x + dropout(linear_out)
x = torch.transpose(x,1, 2)
x = batchNorm2(x)
x = torch.transpose(x,1, 2)

############################# Calculate prediction error #####################
# Test Fit model multiple heads + layers 
from BasicTransformerModel import Transformer

transf_model = Transformer(
    context_size=beforePts+afterPts, 
    context_block=50,
    output_dim=targetPts,
    model_dim=50,
    num_heads = 5,
    num_layers = 4,
    lr=0.001,
    warmup=1,
    max_iters=201,
    dropout=0.2,
    input_dropout=0.2,
    mask = None) 
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
                               patience=25, verbose=False, mode="min")

trainer = pl.Trainer(#logger=neptune_logger,
                     accelerator='auto', #devices=1, # Devices = number of gpus 
                     callbacks=[early_stopping],
                     max_epochs=10,
                     log_every_n_steps=10)

trainer.fit(transf_model, dl_train, dl_val)

# Get loss on validation data
# metric = torch.nn.MSELoss()
# loss = metric(pred, y)
# x, y = next(iter(dl_train))
# pred = transf_model.forward(x)

pred_error = []
iter_dl_val = iter(dl_val)
for _ in range(int(100000/batchSize)):
    x, y = next(iter_dl_val)
    pred = transf_model(x) 
    pred_er = abs(pred-y)
    pred_error.append(torch.mean(pred_er, dim=0).detach().numpy()) # Add mean predicion over samples 

abs_pred_error = list(zip(*pred_error))
abs_pred_error = list(map(lambda x: sum(x)/10, abs_pred_error))

print("Avg L1: ", sum(abs_pred_error)/len(abs_pred_error))

# AVG abs error PRelu: Avg L1:  18.02904111862183
# AVG abs error Relu: Avg L1:  16.37
# AVG abs error GELU: Avg L1:  18.14
# trainer = pl.Trainer(devices="auto",accelerator="auto")

# trainer.fit(transf_model, dl_train, dl_val)

############################## Overlapping patches ###########################
x = torch.arange(0., 5120000)

x=x.reshape(10000,512) #[batch_size = 10, before/afterpts = 6] |[10000, 500]
x.shape

x = x.unfold(dimension = 1, size = 8, step = 4) # batch_size x no. patches x stride
x.shape # if stride < size then shape 
x=x.reshape(10000,-1 ,8)
x.shape