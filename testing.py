# Import raw EEG data 
import mne_bids as mb
# import mne 
import torch
import pytorch_lightning as pl
# import numpy as np 

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch.nn as nn
# import math

import data_utils4 as du
# from LinearModel import linearModel

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

def percentDifference(new, old):
    print((new-old)/old*100)

numbers = {"linear_model": {"MAE": 8.251923561096191, "MSE": 142.62442016601562}, 
"vanilla": {"MAE": 8.341436386108398, "MSE": 140.40516662597656}, 
"L1": {"MAE": 8.226238250732422, "MSE": 137.05406188964844}, 
"LogCosh": {"MAE": 8.302467346191406, "MSE": 139.27798461914062}, 
"overlapping": {"MAE": 8.232880592346191, "MSE": 137.25233459472656}, 
"TUPE-A": {"MAE": 8.22114086151123, "MSE": 137.0876922607422}, 
"TUPE-ALiBi": {"MAE": 8.196436882019043, "MSE": 136.2227325439453}, 
"TUPE-R": {"MAE": 8.174236297607422, "MSE": 135.5532989501953}, 
"ALiBi": {"MAE": 8.297511100769043, "MSE": 138.69561767578125}, 
"CH-Indp": {"MAE": 7.648974895477295, "MSE": 116.88292694091797}}

# TO L1
percentDifference(numbers['L1']['MAE'], numbers['linear_model']['MAE'])
percentDifference(numbers['L1']['MSE'], numbers['linear_model']['MSE'])

#Overlapping
percentDifference(numbers['overlapping']['MAE'], numbers['L1']['MAE'])
percentDifference(numbers['overlapping']['MSE'], numbers['L1']['MSE'])

# Relative TUPE
percentDifference(numbers['TUPE-R']['MAE'], numbers['L1']['MAE'])
percentDifference(numbers['TUPE-R']['MSE'], numbers['L1']['MSE'])

#CH-indp 
percentDifference(numbers['CH-Indp']['MAE'], numbers['TUPE-R']['MAE'])
percentDifference(numbers['CH-Indp']['MSE'], numbers['TUPE-R']['MSE'])

#Best model vs. linear model 
percentDifference(numbers['CH-Indp']['MAE'], numbers['linear_model']['MAE'])
percentDifference(numbers['CH-Indp']['MSE'], numbers['linear_model']['MSE'])


percentDifference(6.421812057495117, 6.4262309074401855)#B
percentDifference(6.44356107711792, 6.376743793487549) #S

percentDifference(80.7712631225586, 80.65196990966797) #B
percentDifference(79.54498291015625, 80.88289642333984) #S


#Data split 
percentDifference(7.61189079284668, 7.648073196411133) 
percentDifference(7.075584411621094, 7.074428558349609) 

# Loss by channel 
percentDifference(5.516302585601807, 8.59657096862793) 
percentDifference(8.824179649353027, 8.59657096862793) 

# Varying target MAE;MSE
percentDifference(7.65, 8.25) ; percentDifference(116.88, 142.62)
percentDifference(9.30, 10.34) ; percentDifference(176.98, 228.80)
percentDifference(10.14, 11.18) ; percentDifference(213.91, 267.56)

# CH-Linear 
percentDifference(7.65, 7.44) ; percentDifference(116.88, 113.50)
percentDifference(8.25, 7.44) ; percentDifference(142.62, 113.50)


############################ Sampling rate ############################
# B x C x P x L5  
t = torch.tensor([[[[1, 1, 1, 1, 1]],
                  [[2, 2, 2, 2, 2]],
                  [[3, 3, 3, 3, 3]]],
                  [[[1, 1, 1, 1, 1]],
                    [[2, 2, 2, 2, 2]],
                    [[3, 3, 3, 3, 3]]]])

t_re = t.reshape(2*3, 1, 5)



pred_error = []
iter_dl_test = iter(dl_train)
for i in range(int(testSize/1000)):
    if i % 10 == 0: print("Calculating Prediction Error for Batch no: ", i)
    x, y = next(iter_dl_test)
    pred = model(x) 
    B, C, NP = y.shape
    print("Shape of pred before: ", pred.shape)
    pred = pred.reshape(B, C, NP)
    print("Shape of pred: ", pred.shape)
    pred_er = abs(pred-y)
    pred_error.append(pred_er.detach()) # Add mean predicion over samples 

abs_pred_error = torch.cat(list(map(lambda x: x.clone().detach(), pred_error)), dim=0)

MSE = torch.mean(torch.mean(torch.square(abs_pred_error), dim=0), dim=1) # Overall 

MAE = torch.mean(abs_pred_error, dim=0)

MAE = torch.mean(MAE, dim=1)


t_re.reshape(2, 3, 1, 5)

import mne
bidsPath = 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
paths=du.returnFilePaths(bidsPath,subjectIds[6:],sessionIds=['001', '002', '003', '004'])

n_times = []
seconds = []

for f in paths:
    tempRaw=mne.io.read_raw_eeglab(f, preload=True,verbose=False)
    n_times.append(tempRaw.n_times)
    seconds.append(tempRaw.times[-1])
    print(tempRaw)
    
print(f'smallet series {min(n_times)}')
tempRaw.info["sfreq"] #250

"It is 7575975 datapoints meaning 30303.9 seconds "
# 8282945 / 33131 # For training data 
8282945*3*56 # length x channels used x files = 1987906800
1987906800*3 # = 5,963,720,400
#1,240,299,060
6250000*(1024+96) # Number samples times length of context size and target
# = 7,000,000,000

tempRaw.drop_channels([0])


"""
Full length: 

tempRaw.n_times
Out[156]: 7575975

7575975/250
Out[157]: 30303.9

30303/60
Out[158]: 505.05
"""
#######################################################################

trainIds=subjectIds.copy()[3:]

for _ in range(5): # Remove some files 
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

print('Loading validation data, subject = ' + subjectIds[valSub])
ds_val=du.EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,
                                 targetPts=targetPts, channelIdxs=1,
                                 preprocess=False,limit=100000,
                                 transform=mytransform)
dl_val=torch.utils.data.DataLoader(ds_val, batch_size=batchSize)

train_iter = iter(dl_train)
inputs, y = next(train_iter)

val_iter = iter(dl_val)
inputs, y = next(val_iter)

# from TUPETransformerModel import TUPEOverlappingTransformer
# transf_model = TUPEOverlappingTransformer(
#     context_size=512+512, 
#     patch_size=64,
#     step=32,
#     output_dim=96,
#     model_dim=64,
#     num_heads=8,
#     num_layers=1,
#     lr=0.001,
#     warmup=1,
#     max_iters=100,
#     dropout=0.2,
#     input_dropout=0.2,
#     mask = None, 
#     only_before=True) 

# early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
#                                patience=25, verbose=False, mode="min")

# trainer = pl.Trainer(#logger=neptune_logger,
#                      accelerator='auto', #devices=1, # Devices = number of gpus 
#                      callbacks=[early_stopping],
#                      max_epochs=3,
#                      log_every_n_steps=10)

# trainer.fit(transf_model, dl_train, dl_val)

# transf_model(inputs)

linearModel = torch.nn.Linear(1024, 96)
linearModel.weight.shape
############################# Calculate prediction error #####################
# AVG abs error PRelu: Avg L1:  18.02904111862183
# AVG abs error Relu: Avg L1:  16.37
# AVG abs error GELU: Avg L1:  18.14
# trainer = pl.Trainer(devices="auto",accelerator="auto")

int((((1024/2)-64)/32+1)*2)

################################## TUPE Relative ####################################
import math
def tupe_product(q, k):
    d = q.size()[-1]
    attn = torch.matmul(q, k.transpose(-2, -1))
    return attn / math.sqrt(2*d)

k.shape
kT = k.transpose(-2, -1)
kT.shape

test = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).reshape((2, 4))
test.transpose(-2, -1)
# In TUPEMultiheadAttention()
inputs, y = next(train_iter)
x1, x2 = inputs
x1 = x1.unfold(dimension = 1, size = 64, 
             step = 32)
x2 = x2.unfold(dimension = 1, size =64, 
             step = 32)
x=torch.cat((x1,x2),dim=1) # shape torch.Size([10000, 30, 64])
# PE = positional_encoding(x)

# __init__
input_dim = 64
embed_dim= 64
num_heads = 16
assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

Wqkv_proj = nn.Linear(input_dim, 3*embed_dim, bias=False) # 3 because we have 3 projection matrices
Wo_proj = nn.Linear(embed_dim, embed_dim, bias=False)
UqUk_proj = nn.Linear(embed_dim, 2*embed_dim, bias=False) 

# Should be added in forward()


batch_size, patch_length, input_dim = x.size()
head_dim = embed_dim // num_heads
# First project x into q, k, and v i.e. multiply 
qkv = Wqkv_proj(x) # torch.Size([10, 16, 64]) -> torch.Size([10, 16, 192])
PE_term = UqUk_proj(PE)

# Separate U_q and U_k from linear output 
PE_term = PE_term.reshape(batch_size, -1, num_heads, 2*head_dim)
# patches are divided across heads to be processed i.e. 16 patches of length 4 are processed on 16 different heads
# Is that correctly understood? 
PE_term = PE_term.permute(0, 2, 1, 3) # [Batch, Head, no patches, head_dim]
Uq, Uk= PE_term.chunk(2, dim=-1)

# Separate Q, K, V from linear output
qkv = qkv.reshape(batch_size, -1, num_heads, 3*head_dim)
# patches are divided across heads to be processed i.e. 16 patches of length 4 are processed on 16 differen heads
# Is that correctly understood? 
qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, no pathces, head_dim]
q, k, v = qkv.chunk(3, dim=-1) # q shape torch.Size([10000, 16, 30, 4])

PE_attn = tupe_product(Uq, Uk) #torch.Size([10000, 16, 30, 30]) 
word_attn = tupe_product(q, k) #torch.Size([10000, 16, 30, 30])

attention = nn.functional.softmax(PE_attn + word_attn + PE_r, dim=-1)


############################## ALIBI #####################################
from ALiBiTUPETransformerModel import ALiBiTransformer

cols = torch.arange(no_patches, dtype=torch.long)[:, None]
rows = torch.arange(no_patches, dtype=torch.long)[None, :]
relative_position = rows - cols 

RP_heads = relative_position[None,:,:].expand(n_heads, -1, -1)
ratio = 8/n_heads
scalars = torch.tensor([1/2**i for i in np.arange(ratio, 8+ratio, ratio)], dtype=torch.float32)

RP_heads = scalars[:, None, None] * RP_heads

PE_r = PE_r[None,:, :,:].expand(batch_size,-1, -1, -1)

trans_model = ALiBiTransformer(
    context_size=512+512, 
    patch_size=64,
    step = 32,
    output_dim=96,
    model_dim=64,
    num_heads = 16,
    num_layers = 3,
    lr=0.001,
    warmup=1,
    max_iters=100,
    dropout=0.2,
    input_dropout=0.2,
    mask = None,
    TUPE=True)

train_iter = iter(dl_train)
inputs, y = next(train_iter)
trans_model(inputs)

batch = next(train_iter)
trans_model.configure_optimizers()
trans_model.training_step(batch, 1)

trans_model.validation_step(batch,1)

trans_model.transformer.layers[0].TUPE_attn.UqUk_proj.weight.shape

######
from RelativeTUPETransformerModel import RelativeTUPETransformer

trans_model=RelativeTUPETransformer(
    context_size=512+512, 
    patch_size=64,
    step = 64,
    output_dim=96,
    model_dim=64*2,
    num_heads = 16,
    num_layers = 3,
    lr=0.001,
    warmup=1,
    max_iters=100,
    dropout=0.2,
    input_dropout=0.2,
    mask = None)


sum(p.numel() for p in trans_model.R_PE.parameters() if p.requires_grad)
#Different for each head= 
1024/64

inputs, y = next(train_iter)

trans_model(inputs)

batch = next(train_iter)

trans_model.configure_optimizers()
trans_model.training_step(batch, 1)
trans_model.validation_step(batch,1)

########################## CHANNEL INDEPENDENCE ##############################
def mytransform(raw):
    raw.filter(0.1,40)
    raw._data=raw._data*1e6
    return raw

pl.seed_everything(42, workers=True)

batchSize= 1000
channelIdxs=[1,19,23] 
valSub=0
beforePts=512
afterPts=512
targetPts=96

bidsPath= 'Y:\\NTdata\\BIDS\\EESM17\\'
# bidsPath = 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
subjectIds=mb.get_entity_vals(bidsPath,'subject', with_key=False)
trainIds=subjectIds.copy()
trainIds.pop(valSub)

for _ in range(8): # Remove some files 
    trainIds.pop(0)


flattenOutSize= int((((512*2/2)-64)/32+1)*2*64*2)


import data_utils_channelIndp as duCH

trainPaths=duCH.returnFilePaths(bidsPath,trainIds,sessionIds=['001']) # There is onlyone session in small dataset
valPaths=duCH.returnFilePaths(bidsPath,[subjectIds[valSub]],sessionIds=['001'])

print('Loading training data')
ds_train_CHI =duCH.EEG_dataset_from_paths(trainPaths, beforePts=beforePts,
                                   afterPts=afterPts,targetPts=targetPts, 
                                   channelIdxs=channelIdxs,preprocess=False,
                                    limit=None,train_size = 1000,
                                   transform=None
                                   )
dl_train=torch.utils.data.DataLoader(ds_train_CHI, batch_size=10000, shuffle=True)

print('Loading validation data, subject = ' + subjectIds[valSub])
ds_val_CHI = duCH.EEG_dataset_from_paths(trainPaths, beforePts=beforePts,afterPts=afterPts,
                                 targetPts=targetPts, channelIdxs=channelIdxs,
                                 preprocess=False,limit=100000,
                                 transform=mytransform)
dl_val=torch.utils.data.DataLoader(ds_val_CHI, batch_size=3333)

train_iter = iter(dl_train)
inputs, y = next(train_iter)

x1, x2 = inputs
x_l = torch.cat((x1, x2),dim=2)
B, C, L = x_l.shape
x_l = x_l.reshape(B*C, L)

from ChannelLinearModel import ChiIndLinearTransformer

CH_model = ChiIndLinearTransformer(
    context_size=512+512, 
    patch_size=64,
    step = 64,
    output_dim=96,
    model_dim=64*2,
    num_heads = 16,
    num_layers = 3,
    lr=0.001,
    warmup=1,
    max_iters=100,
    dropout=0.2,
    input_dropout=0.2,
    mask = None,
    only_before=False) 

CH_model(inputs)

x1 = x1.unfold(dimension = 2, size = 64, 
             step = 32)
x2 = x2.unfold(dimension = 2, size = 64, 
             step = 32)
x=torch.cat((x1,x2),dim=2)

######################### MODEL SIZE #####################################
def countNonEmbeddingParameters(model):
    input_net = sum(p.numel() for p in model.input_net.parameters() if p.requires_grad)
    R_PE = sum(p.numel() for p in model.R_PE.parameters() if p.requires_grad)
    print("Embedding: ", input_net+R_PE)
    
    transformer_param = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    output_net_param = sum(p.numel() for p in model.output_net.parameters() if p.requires_grad)
    print("Non-embedding: ", transformer_param + output_net_param)

from ChannelIndpTransformerModel import ChiIndTUPEOverlappingTransformer
CH_model = ChiIndTUPEOverlappingTransformer(
    context_size=2048+2048, 
    patch_size=64,
    step = 64,
    output_dim=96,
    model_dim=64*2,
    num_heads = 16,
    num_layers = 3,
    lr=0.001,
    warmup=1,
    max_iters=100,
    dropout=0.2,
    input_dropout=0.2,
    mask = None,
    only_before=False) 

countNonEmbeddingParameters(CH_model)
CH_model.patches

"""
0 | metric              | L1Loss               | 0     
1 | input_net           | Sequential           | 4.2 K 
2 | positional_encoding | PositionalEncoding   | 0     
3 | R_PE                | RelativePositionBias | 512   
4 | transformer         | TransformerEncoder   | 123 K 
5 | output_net          | Sequential           | 117 K 
-------------------------------------------------------
"""

early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
                               patience=25, verbose=False, mode="min")

trainer = pl.Trainer(#logger=neptune_logger,
                     accelerator='auto', #devices=1, # Devices = number of gpus 
                     callbacks=[early_stopping],
                     max_epochs=1,
                     log_every_n_steps=10)

trainer.fit(CH_model, dl_train, dl_train)



