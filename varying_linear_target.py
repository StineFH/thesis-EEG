import os
import mne_bids as mb
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import data_utils4 as du

from LinearModel import linearModel

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


def runCurrentModel(dl_train, dl_val, target):
    ######################## Make Neptune Logger ############################
    #https://docs.neptune.ai/api/neptune/#init_run
    NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZTFjOGZiMS01NDFjLTRlMzktOTBiYS0yNDcxM2UzNWM2ZTYifQ=='

    neptune_logger = pl.loggers.NeptuneLogger(
        api_key = NEPTUNE_API_TOKEN,
        project="stinefh/thesis-EEG", 
        source_files=["varying_linear_target.py", 
                      "data_utils_channelIndp.py", 
                      "ChannelIndpTransformerModel.py"],
        capture_hardware_metrics=False,
        capture_stdout=False,
        capture_stderr=False,
        capture_traceback=False
        # tags=neptuneTags
        )
    neptune_logger.log_hyperparams({'valSub':['004', '005', '006']})
    neptune_logger.log_hyperparams({'trainSub':['007','020']})
    neptune_logger.log_hyperparams({'beforePts':512, 'afterPts':512})
    neptune_logger.log_hyperparams({'lr schedular':"CosineWarmup"})
    
    ################## make Model, Earlystop, Trainer and Fit #################
    lin_model = linearModel(0.001,
                            512+512, 
                            target, 
                            warmup=6250, 
                            max_iters=188000) 
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                   patience=25, verbose=False, mode="min")
    
    trainer = pl.Trainer(logger=neptune_logger,
                         accelerator='gpu', devices=2, # Devices = number of gpus 
                         callbacks=[early_stopping],
                         max_epochs=60,
                         log_every_n_steps=50)
    
    trainer.fit(lin_model, dl_train, dl_val)
    
    # Save best model
    torch.save(lin_model.state_dict(), 'linear_model_snapshot/' + neptune_logger.version + '.pt')
    
    # Calculate average absolute prediction error 
    lin_model.load_state_dict(torch.load("./linear_model_snapshot/" + neptune_logger.version + '.pt'))
    pred_error = []
    iter_dl_val = iter(dl_val)
    for _ in range(int(limit_val/batchSize)):
        x, y = next(iter_dl_val)
        pred = lin_model(x) 
        pred_er = abs(pred-y)
        pred_error.append(pred_er.detach()) # Add mean predicion over samples 
    
    # Calculate validation loss 
    abs_pred_error = torch.cat(list(map(torch.tensor, pred_error)), dim=0)
    MSE = torch.mean(torch.mean(torch.square(abs_pred_error), dim=0)) # Overall 
    MAE = torch.mean(abs_pred_error, dim=0)
    
    torch.save(MAE, './lin_model_prediction_error/' + 'MAE_' + neptune_logger.version + '.pt')
    
    neptune_logger.finalize('Success')
    neptune_logger.experiment.stop()
    
    return {'MAE': float((sum(MAE)/len(MAE)).detach().numpy()), 'MSE': float(MSE.detach().numpy())}   



if __name__ == '__main__':
    
    sessionIds = ['001', '002', '003', '004']
    channelIdxs=[1,19,23]
    targetPts=96
    
    batchSize = 10000
    train_size=6250000
    limit_val=1875000
    
    outputs = {} 
    targets = [192, 336] # The one on 96 is THES-71
    
    for target in targets:
        dl_train, dl_val = getData(sessionIds, channelIdxs, 512, 512, 
                                   target, batchSize, train_size, limit_val)
        
        MAE_MSE = runCurrentModel(dl_train, dl_val, target)
        
        outputs[str(target)] = MAE_MSE
        print('Val loss for', target, ': ', outputs[str(target)])
    
        
    torch.save(outputs, './test_plots/' + 'validation_loss_L_target'+ '.pt')
    
