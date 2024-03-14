import os
import mne_bids as mb
import torch

import data_utils4 as du
from LinearModel import linearModel
from TransformerModel import Transformer

cuda=torch.device('cpu')

if torch.cuda.is_available():
    cuda=torch.device('cuda:0')
    print(torch.version.cuda)
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f/1e6)
else:
    print("no cuda available")
    
    
def getAbsPredictionError(
        model,
        model_name,
        model_path,
        error_path,
        batchSize= 10000,
        channelIdxs=[1,19,23], 
        valSub=0,
        beforePts=500,
        afterPts=500,
        targetPts=100,
        sessionIds = ['001', '002'], # i.e. only half the data in EESM19
        limit_val = 100000 # Dataset size 
        ):

    tempPath= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
    if os.path.isdir(tempPath):
        bidsPath = tempPath
        
    subjectIds=mb.get_entity_vals(bidsPath,'subject', with_key=False)
    trainIds=subjectIds.copy()
    trainIds.pop(valSub)
    
    valPaths=du.returnFilePaths(bidsPath,[subjectIds[valSub]],sessionIds=sessionIds)
    
    def mytransform(raw):
        raw.filter(0.1, 40)
        raw._data=raw._data*1e6
        return raw
    
    print('Loading validation data, subject = ' + subjectIds[valSub])
    ds_val=du.EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,
                                     targetPts=targetPts, channelIdxs=1,
                                     preprocess=False,limit=limit_val,transform=mytransform
                                     )
    dl_val=torch.utils.data.DataLoader(ds_val, 
                                       # num_workers=8,
                                       batch_size=batchSize
                                       )
    
    model.load_state_dict(torch.load(model_path + model_name))
    pred_error = []
    iter_dl_val = iter(dl_val)
    for _ in range(int(100000/batchSize)):
        x, y = next(iter_dl_val)
        
        pred = model(x) 
        pred_er = abs(pred-y)
        pred_error.append(torch.mean(pred_er, dim=0).detach().numpy()) # Add mean predicion over samples 
    
    torch.save(pred_error, error_path + model_name)

###############################################################################
model_name = 'THES-34.pt'
model_path = './linear_model_snapshot/'
error_path = './lin_model_prediction_error/'
# model_path = './transformer_model_snapshot/'
# error_path = './transformer_prediction_error/'

lin_model = linearModel(lr=0.001,input_size=500+500, output_size=100, 
                        warmup=300,
                        max_iters=3300) 

# transf_model = Transformer(
#     context_size=500+500, 
#     context_block=50,
#     output_dim=100,
#     model_dim=50,
#     num_heads=1,
#     num_layers=1,
#     lr=0.001,
#     warmup=300,
#     max_iters=3300,
#     dropout=0.0,
#     input_dropout=0.0,
#     mask = None) 

getAbsPredictionError(
    lin_model,
    model_name=model_name,
    model_path = model_path,
    error_path = error_path,
    batchSize= 10000,
    channelIdxs=[1,19,23], 
    valSub=0,
    beforePts=500,
    afterPts=500,
    targetPts=100,
    sessionIds = ['001', '002'], # i.e. only half the data in EESM19
    limit_val = 100000 # Dataset size 
    )

