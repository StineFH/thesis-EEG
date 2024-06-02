import os
import mne_bids as mb
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import data_utils_channelIndp as du

from ChannelIndpTransformerModel import ChiIndTUPEOverlappingTransformer

pl.seed_everything(42, workers=True)



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
    
    
    
def getData(sessionIds, channelIdxs, beforePts, afterPts, targetPts, 
            batchSize, train_size, limit_val):
    def mytransform(raw):
        raw.filter(0.1, 40)
        raw._data=raw._data*1e6
        return raw
    
    # where are we?:
    tempPath= '/data/'
    if os.path.isdir(tempPath):
        bidsPath = tempPath
        
    subjectIds=mb.get_entity_vals(bidsPath,'subject', with_key=False)
    trainIds=subjectIds.copy()[6:]
    valIds = subjectIds.copy()[3:6]

    trainPaths=du.returnFilePaths(bidsPath,trainIds,sessionIds=sessionIds) # There is onlyone session in small dataset
    valPaths=du.returnFilePaths(bidsPath,valIds,sessionIds=sessionIds)
    
    print('Loading training data')
    ds_train=du.EEG_dataset_from_paths(trainPaths, beforePts=beforePts,
                                       afterPts=afterPts,targetPts=targetPts, 
                                       channelIdxs=channelIdxs,preprocess=False,
                                        limit=None,train_size = train_size,
                                        transform=mytransform
                                       )
    dl_train=torch.utils.data.DataLoader(ds_train, batch_size=batchSize, 
                                         shuffle=True, num_workers=16)
    
    print('Loading validation data, subject = ' + str(valIds))
    ds_val=du.EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,
                                     targetPts=targetPts, channelIdxs=channelIdxs,
                                     preprocess=False,limit=limit_val,transform=mytransform
                                     )
    dl_val=torch.utils.data.DataLoader(ds_val, batch_size=batchSize,
                                       num_workers=16)
    
    return dl_train, dl_val


def runCurrentModel(model_dims, n_layers, dl_train, dl_val):
    ######################## Make Neptune Logger ############################
    #https://docs.neptune.ai/api/neptune/#init_run
    NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZTFjOGZiMS01NDFjLTRlMzktOTBiYS0yNDcxM2UzNWM2ZTYifQ=='

    neptune_logger = pl.loggers.NeptuneLogger(
        api_key = NEPTUNE_API_TOKEN,
        project="stinefh/thesis-EEG", 
        source_files=["run_CHIndTUPETransformer.py", 
                      "data_utils_channelIndp.py", 
                      "ChannelIndpTransformerModel.py"],
        capture_hardware_metrics=False,
        capture_stdout=False, # Do not log these things -> since it causes error sometimes 
        capture_stderr=False,
        capture_traceback=False
        # tags=neptuneTags
        )
    neptune_logger.log_hyperparams({'valSub':['004', '005', '006']})
    neptune_logger.log_hyperparams({'trainSub':['007','020']})
    neptune_logger.log_hyperparams({'beforePts':512, 'afterPts':512})
    neptune_logger.log_hyperparams({'lr schedular':"CosineWarmup"})
    
    ################## make Model, Earlystop, Trainer and Fit #################
    transf_model = ChiIndTUPEOverlappingTransformer(
        context_size=512+512, 
        patch_size=64,
        step = 64,
        output_dim=96,
        model_dim=model_dims,
        num_heads = 16,
        num_layers = n_layers,
        lr=0.001,
        warmup=6250*2, # Double to make same number epochs as other
        max_iters=188000*2,
        dropout=0.2,
        input_dropout=0.2,
        only_before=False) 
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                   patience=25, verbose=False, mode="min")
    
    trainer = pl.Trainer(logger=neptune_logger,
                         accelerator='gpu', devices=2, # Devices = number of gpus 
                         callbacks=[early_stopping],
                         max_epochs=300,
                         log_every_n_steps=50)
    
    trainer.fit(transf_model, dl_train, dl_val)
    
    # Save best model
    torch.save(transf_model.state_dict(), 'transformer_model_snapshot/' + neptune_logger.version + '.pt')
    
    # Calculate average absolute prediction error 
    transf_model.load_state_dict(torch.load("./transformer_model_snapshot/" + neptune_logger.version + '.pt'))
    pred_error = []
    iter_dl_val = iter(dl_val)
    for _ in range(int(limit_val/batchSize)):
        x, y = next(iter_dl_val)
        pred = transf_model(x) 
        B, C, NP = y.shape
        y = y.reshape(B*C, NP)
        pred_er = abs(pred-y)
        pred_error.append(pred_er.detach()) # Add mean predicion over samples 
    
    torch.save(pred_error, './transformer_prediction_error/' + neptune_logger.version + '.pt')
    
    # Calculate validation loss 
    abs_pred_error = torch.cat(list(map(torch.tensor, pred_error)), dim=0)
    MSE = torch.mean(torch.mean(torch.square(abs_pred_error), dim=0)) # Overall 
    MAE = torch.mean(abs_pred_error, dim=0)
    
    neptune_logger.finalize('Success')
    neptune_logger.experiment.stop()
    
    return {'MAE': float((sum(MAE)/len(MAE)).detach().numpy()), 'MSE': float(MSE.detach().numpy())}   



if __name__ == '__main__':
    
    sessionIds = ['001', '002', '003', '004']
    channelIdxs=[1,19,23]
    beforePts = 512
    afterPts=512
    targetPts=96
    
    batchSize = 3333
    train_size=2083333*2
    limit_val=625000
    
    dl_train, dl_val = getData(sessionIds, channelIdxs, beforePts, afterPts, 
                               targetPts,batchSize, train_size, limit_val)
    
    outputs = {}
    model_dim = [16, 64, 64*2, 64*3, 64*4, 64*7]
    layers = [1, 2, 3, 3*2, 3*5, 3*8]
    
    assert len(model_dim) == len(layers), "model_dim and layers have different lengths"

    for d_model, layer in zip(model_dim, layers):
        MAE_MSE = runCurrentModel(d_model, layer, dl_train, dl_val)
        
        outputs[str(d_model)] = MAE_MSE
        print('Val loss for', d_model, ': ', outputs[str(d_model)])
    
        
    torch.save(outputs, './test_plots/' + 'validation_loss_model_sizes'+ '.pt')

    
"""

Val loss for 16 :  {'MAE': 6.687623500823975, 'MSE': 86.76795196533203}
Val loss for 64 :  {'MAE': 6.447359561920166, 'MSE': 81.32015991210938}
Val loss for 128 :  {'MAE': 6.5262627601623535, 'MSE': 82.8070297241211}
Val loss for 192 :  {'MAE': 6.903939723968506, 'MSE': 91.82431030273438}

"""