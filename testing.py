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

############################# Calculate prediction error #####################
# AVG abs error PRelu: Avg L1:  18.02904111862183
# AVG abs error Relu: Avg L1:  16.37
# AVG abs error GELU: Avg L1:  18.14
# trainer = pl.Trainer(devices="auto",accelerator="auto")

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
    TUPE=False)

trans_model(inputs)


from RelativeTUPETransformerModel import RelativeTUPETransformer

trans_model=RelativeTUPETransformer(
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
    mask = None)

trans_model(inputs)

########################## CHANNEL INDEPENDENCE ##############################
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

# bidsPath= 'Y:\\NTdata\\BIDS\\EESM17\\'
bidsPath = 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
subjectIds=mb.get_entity_vals(bidsPath,'subject', with_key=False)
trainIds=subjectIds.copy()
trainIds.pop(valSub)

for _ in range(17): # Remove some files 
    trainIds.pop(0)

trainPaths=du.returnFilePaths(bidsPath,trainIds,sessionIds=['001']) # There is onlyone session in small dataset
valPaths=du.returnFilePaths(bidsPath,[subjectIds[valSub]],sessionIds=['001'])

print('Loading training data')
ds_train_CHI =du.EEG_dataset_from_paths(trainPaths, beforePts=beforePts,
                                   afterPts=afterPts,targetPts=targetPts, 
                                   channelIdxs=channelIdxs,preprocess=False,
                                    limit=None,train_size = 100000,
                                   transform=mytransform)
dl_train=torch.utils.data.DataLoader(ds_train_CHI, batch_size=batchSize, shuffle=True)

print('Loading validation data, subject = ' + subjectIds[valSub])
ds_val_CHI = du.EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,
                                 targetPts=targetPts, channelIdxs=channelIdxs,
                                 preprocess=False,limit=100000,
                                 transform=mytransform)
dl_val=torch.utils.data.DataLoader(ds_val_CHI, batch_size=batchSize)

val_iter = iter(dl_val)
inputs, y = next(val_iter)

x1, x2 = inputs

x = x1.unfold(dimension = 1, size = 32,
              step = 16) #31 wtih length 32
x.shape
from TUPETransformerModel import TUPEOverlappingTransformer

transf_model = TUPEOverlappingTransformer(
    context_size=512+512, 
    patch_size=32,
    step = 16,
    output_dim=96,
    model_dim=32,
    num_heads = 16,
    num_layers = 3,
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

# q size torch.Size([10000, 16, 62, 2])
# size of PE_attn:  torch.Size([10000, 16, 62, 62]) #Batch_size x  
# size of word_attn:  torch.Size([10000, 16, 62, 62])