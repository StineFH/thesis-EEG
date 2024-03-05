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
        targetPts=100):

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
    trainIds=subjectIds.copy()
    trainIds.pop(valSub)
    trainPaths=du.returnFilePaths(bidsPath,trainIds,sessionIds=['001', '002']) # There is onlyone session in small dataset
    valPaths=du.returnFilePaths(bidsPath,[subjectIds[valSub]],sessionIds=['001', '002'])
    
    
    print('Loading training data')
    ds_train=du.EEG_dataset_from_paths(trainPaths, beforePts=beforePts,
                                       afterPts=afterPts,targetPts=targetPts, 
                                       channelIdxs=channelIdxs,preprocess=False,
                                       limit=None #,transform=mytransform
                                       )
    dl_train=torch.utils.data.DataLoader(ds_train, batch_size=batchSize, 
                                         shuffle=True, num_workers=8)
    
    print('Loading validation data, subject = ' + subjectIds[valSub])
    ds_val=du.EEG_dataset_from_paths(valPaths, beforePts=beforePts,afterPts=afterPts,
                                     targetPts=targetPts, channelIdxs=1,
                                     preprocess=False,limit=100000 #,transform=mytransform
                                     )
    dl_val=torch.utils.data.DataLoader(ds_val, batch_size=batchSize,
                                       num_workers=8)
    
    ######################## Make Neptune Logger ############################
    #https://docs.neptune.ai/api/neptune/#init_run
    
    # neptune_token=os.getenv('NEPTUNE_API_TOKEN')

    neptune_logger = pl.loggers.NeptuneLogger(
        project="stinefh/thesis-EEG", 
        source_files=["benchmark_linear_model.py", 
                      "data_utils4.py", 
                      "LinearModel.py"]
        # tags=neptuneTags
        )
    neptune_logger.log_hyperparams({'valSub':subjectIds[valSub]})
    neptune_logger.log_hyperparams({'trainSub':trainIds})
    neptune_logger.log_hyperparams({'beforePts':beforePts, 'afterPts':afterPts})
    neptune_logger.log_hyperparams({'lr schedular':"CosineWarmup"})
    
    ################## make Model, Earlystop, Trainer and Fit #################
    lin_model = linearModel(0.001,beforePts+afterPts, targetPts, warmup=300, 
                            max_iters=15000) 
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                   patience=25, verbose=False, mode="min")
    
    trainer = pl.Trainer(logger=neptune_logger,
                         devices="auto",
                         accelerator="auto", 
                         callbacks=[early_stopping],
                         max_epochs=1000,# 1000
                         log_every_n_steps=50,
                         gpus=1
                         )
    
    trainer.fit(lin_model, dl_train, dl_val)
    
    torch.save(lin_model.state_dict(), 'linear_model_snapshot/' + neptune_logger.version + '.pt')
    
    neptune_logger.finalize('Success')
    neptune_logger.experiment.stop()
    
    return trainer, lin_model

################################ Run Experiment ###############################

targetPts=100
beforePts=500
afterPts=500

trainer,net=runExperiment(batchSize= 10000,
                          channelIdxs=[1,19,23],
                          valSub=0, 
                          targetPts=targetPts,
                          beforePts=beforePts, 
                          afterPts=afterPts)

