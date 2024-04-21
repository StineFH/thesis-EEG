# Import raw EEG data 
import mne_bids as mb
import mne 
import torch
import pytorch_lightning as pl
import numpy as np 

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch.nn as nn
import math

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
beforePts=512
afterPts=512
targetPts=96

bidsPath= 'Y:\\NTdata\\BIDS\\EESM17\\'
subjectIds=mb.get_entity_vals(bidsPath,'subject', with_key=False)
trainIds=subjectIds.copy()
trainIds.pop(valSub)

for _ in range(6): # Remove some files 
    trainIds.pop(0)

trainPaths=du.returnFilePaths(bidsPath,trainIds,sessionIds=['001']) # There is onlyone session in small dataset
valPaths=du.returnFilePaths(bidsPath,[subjectIds[valSub]],sessionIds=['001'])

print('Loading training data')
ds_train =du.EEG_dataset_from_paths(trainPaths, beforePts=beforePts,
                                   afterPts=afterPts,targetPts=targetPts, 
                                   channelIdxs=channelIdxs,preprocess=False,
                                    limit=None,train_size = 100000,
                                   transform=mytransform)
dl_train =torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True)

train_iter = iter(dl_train)
inputs, y = next(train_iter)

print('Loading validation data, subject = ' + subjectIds[valSub])
ds_val=du.EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,
                                 targetPts=targetPts, channelIdxs=1,
                                 preprocess=False,limit=100000,
                                 transform=mytransform)
dl_val=torch.utils.data.DataLoader(ds_val, batch_size=batchSize)

val_iter = iter(dl_val)
inputs, y = next(val_iter)

from TUPETransformerModel import TUPEOverlappingTransformer
transf_model = TUPEOverlappingTransformer(
    context_size=512+512, 
    patch_size=32,
    step=16,
    output_dim=96,
    model_dim=32,
    num_heads=8,
    num_layers=1,
    lr=0.001,
    warmup=1,
    max_iters=100,
    dropout=0.2,
    input_dropout=0.2,
    mask = None, 
    only_before=False) 

early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
                               patience=25, verbose=False, mode="min")

trainer = pl.Trainer(#logger=neptune_logger,
                     accelerator='auto', #devices=1, # Devices = number of gpus 
                     callbacks=[early_stopping],
                     max_epochs=3,
                     log_every_n_steps=10)

trainer.fit(transf_model, dl_train, dl_val)

transf_model(inputs)

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

######################## LAYERNORM ###################################
N, C, H = 10000, 127, 64
# z = torch.randn(N, C, H, W)

layer_norm = nn.LayerNorm(64)

output = layer_norm(x)

layer_norm.weight
layer_norm.bias

x.mean((-2, -1)).shape
x.sd((-2, -1)).shape
torch.std(x, dim=(-2, -1))

########################## CHANNEL INDEPENDENCE ##############################
import data_utils_channelIndp as duCI

def mytransform(raw):
    raw.filter(0.1,40)
    raw._data=raw._data*1e6
    return raw

pl.seed_everything(42, workers=True)

batchSize= 1000
channelIdxs=[1,19,23] 
valSub=0
beforePts=512*2
afterPts=0
targetPts=96

bidsPath= 'Y:\\NTdata\\BIDS\\EESM17\\'
subjectIds=mb.get_entity_vals(bidsPath,'subject', with_key=False)
trainIds=subjectIds.copy()
trainIds.pop(valSub)

for _ in range(7): # Remove some files 
    trainIds.pop(0)

trainPaths=duCI.returnFilePaths(bidsPath,trainIds,sessionIds=['001']) # There is onlyone session in small dataset
valPaths=duCI.returnFilePaths(bidsPath,[subjectIds[valSub]],sessionIds=['001'])

print('Loading training data')
ds_train_CHI =duCI.EEG_dataset_from_paths(trainPaths, beforePts=beforePts,
                                   afterPts=afterPts,targetPts=targetPts, 
                                   channelIdxs=channelIdxs,preprocess=False,
                                    limit=None,train_size = 100000,
                                   transform=mytransform)
dl_train=torch.utils.data.DataLoader(ds_train_CHI, batch_size=batchSize, shuffle=True)

print('Loading validation data, subject = ' + subjectIds[valSub])
ds_val_CHI = duCI.EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,
                                 targetPts=targetPts, channelIdxs=channelIdxs,
                                 preprocess=False,limit=100000,
                                 transform=mytransform)
dl_val=torch.utils.data.DataLoader(ds_val_CHI, batch_size=batchSize)

val_iter = iter(dl_val)
inputs, y = next(val_iter)
x = inputs.unfold(dimension = 2, size = patch_length, 
             step = step)


from ChannelIndpTransformerModel import ChiIndTUPEOverlappingTransformer

transf_model = ChiIndTUPEOverlappingTransformer(
    context_size=512+512, 
    patch_size=32,
    step=16,
    output_dim=96,
    model_dim=32,
    num_heads=16,
    num_layers=3,
    lr=0.001,
    warmup=1,
    max_iters=100,
    dropout=0.2,
    input_dropout=0.2,
    mask = None, 
    only_before=True)

early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
                               patience=25, verbose=False, mode="min")

trainer = pl.Trainer(#logger=neptune_logger,
                     accelerator='auto', #devices=1, # Devices = number of gpus 
                     callbacks=[early_stopping],
                     max_epochs=3,
                     log_every_n_steps=10)

trainer.fit(transf_model, dl_train, dl_val)

transf_model(inputs)

