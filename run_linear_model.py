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
    


def runExperiment(
        batchSize= 10000,
        channelIdxs=[1,19,23], 
        valSub=0,
        beforePts=500,
        afterPts=500,
        targetPts=100,
        sessionIds = ['001', '002'], # i.e. only half the data in EESM19
        limit_val = 100000, # Dataset size 
        train_size = 300000,
        max_iters = 15000,
        max_epochs = 1000,
        warmup = 300):

    ####################### Make Datset and DataLoader ########################
    # simple scaling of input (to make it microvolt):
    # transform = lambda x: x*1e6
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
    
    ######################## Make Neptune Logger ############################
    #https://docs.neptune.ai/api/neptune/#init_run
    
    NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZTFjOGZiMS01NDFjLTRlMzktOTBiYS0yNDcxM2UzNWM2ZTYifQ=='

    neptune_logger = pl.loggers.NeptuneLogger(
        api_key = NEPTUNE_API_TOKEN,
        project="stinefh/thesis-EEG", 
        source_files=["benchmark_linear_model.py", 
                      "data_utils4.py", 
                      "LinearModel.py"],
        capture_hardware_metrics=False,
        capture_stdout=False,
        capture_stderr=False,
        capture_traceback=False
        # tags=neptuneTags
        )
    neptune_logger.log_hyperparams({'valSub':subjectIds[valSub]})
    neptune_logger.log_hyperparams({'trainSub':trainIds})
    neptune_logger.log_hyperparams({'beforePts':beforePts, 'afterPts':afterPts})
    neptune_logger.log_hyperparams({'lr schedular':"CosineWarmup"})
    
    ################## make Model, Earlystop, Trainer and Fit #################
    lin_model = linearModel(0.001,
                            beforePts+afterPts, 
                            targetPts, 
                            warmup=warmup, 
                            max_iters=max_iters) 
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                   patience=25, verbose=False, mode="min")
    
    trainer = pl.Trainer(logger=neptune_logger,
                         accelerator='gpu', devices=2, # Devices = number of gpus 
                         callbacks=[early_stopping],
                         max_epochs=max_epochs,
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
        pred_error.append(pred_er.detach().numpy()) # Add mean predicion over samples 
    
    torch.save(pred_error, './lin_model_prediction_error/' + neptune_logger.version + '.pt')
    
    neptune_logger.finalize('Success')
    neptune_logger.experiment.stop()
    
    return trainer, lin_model

################################ Run Experiment ###############################

targetPts=96
beforePts=512
afterPts=512

sessionIds = ['001', '002', '003', '004'] 
limit = 1875000 # validation dataset size
train_size = 6250000 # train dataset size 
batchSize= 10000
channelIdxs=[1,19,23]
valSub=0
max_iters = 188000
max_epochs = 300
warmup = 6250


trainer,net=runExperiment(batchSize= batchSize,
                          channelIdxs=channelIdxs,
                          valSub=valSub, 
                          targetPts=targetPts,
                          beforePts=beforePts, 
                          afterPts=afterPts,
                          sessionIds = sessionIds, # i.e. only half the data in EESM19
                          limit_val = limit, # Dataset size 
                          train_size = train_size,
                          max_iters = max_iters,
                          max_epochs = max_epochs,
                          warmup = warmup)

