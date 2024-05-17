import torch
import data_utils4 as du
import data_utils_channelIndp as CHdu
import mne_bids as mb
import json

############################### Import test data ##############################

def getData(testSize, path, 
            beforePts, afterPts, targetPts, channelIds, sessionIds,
            CH = False):
    def mytransform(raw):
        raw.filter(0.1, 40)
        raw._data=raw._data*1e6
        return raw
    
    subjectIds=mb.get_entity_vals(path,'subject', with_key=False)
    testIds=subjectIds.copy()[:3]
    sessionIds = ['001', '002', '003', '004']
    
    if CH:
        subPath = CHdu.returnFilePaths(path, testIds, sessionIds=sessionIds)
        ds_test = CHdu.EEG_dataset_from_paths(subPath, 
                                            beforePts=beforePts,afterPts=afterPts, 
                                            targetPts=targetPts, channelIdxs=channelIds,
                                            preprocess=False,limit=testSize,
                                            transform=mytransform
                                            )
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=3333, 
                                              shuffle=False, num_workers=16)
    else: 
        subPath = du.returnFilePaths(path, testIds, sessionIds=sessionIds)
        ds_test = du.EEG_dataset_from_paths(subPath, 
                                            beforePts=beforePts,afterPts=afterPts, 
                                            targetPts=targetPts, channelIdxs=channelIds,
                                            preprocess=False,limit=testSize,
                                            transform=mytransform
                                            )
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=3333, shuffle=False,
                                              num_workers=16)
    return ds_test, dl_test

############################### Get MSE and MAE ###############################


def getTestResults(model_d, n_layer, beforePts, afterPts, targetPts, 
                   models_to_run, neptune_names,
                   CH_ds_test=None, CH_dl_test=None,ds_test=None, dl_test=None):
    MAE_MSE_ALL = {}    
    
    for r in range(10):
        print("This is round: ", r)
        
        channelIds=[1,19,23]
        testSize = 625000
        sessionIds = ['001', '002', '003', '004']
        
        CH_ds_test, CH_dl_test = getData(testSize, path, beforePts, afterPts, 
                                         targetPts,channelIds, sessionIds, 
                                         CH=True)
        
        MAE_MSE = {}
        for i, (m, n) in enumerate(zip(models_to_run, neptune_names)):
            if m == "linear_model":
                model_path = './linear_model_snapshot/'
            else:
                model_path = './transformer_model_snapshot/'
            
            warmup=6250
            max_iters=188000
            
            if m == "linear_model":
                from LinearModel import linearModel
                model = linearModel(0.001,
                                        beforePts+afterPts, 
                                        targetPts, 
                                        warmup=warmup, 
                                        max_iters=max_iters) 
            
            elif m == 'vanilla':
                # Vanilla transformer 
                from BasicTransformerModel import Transformer
                model = Transformer(
                    context_size=beforePts+afterPts, 
                    context_block=64,
                    output_dim=targetPts,
                    model_dim=64*2,
                    num_heads = 16,
                    num_layers = 3,
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None) 
            elif m == 'L1':
                from L1LossTransformerModel import Transformer
                model = Transformer(
                    context_size=beforePts+afterPts, 
                    context_block=64,
                    output_dim=targetPts,
                    model_dim=64*2,
                    num_heads = 16,
                    num_layers = 3,
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None) 
            elif m == 'LogCosh':
                from LogCoshLossTransformerModel import Transformer
                model = Transformer(
                    context_size=beforePts+afterPts, 
                    context_block=64,
                    output_dim=targetPts,
                    model_dim=64*2,
                    num_heads = 16,
                    num_layers = 3,
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None) 
            elif m == 'overlapping':
                from overlappingPatchesTransformer import OverlappingTransformer
                model = OverlappingTransformer(
                    context_size=beforePts+afterPts, 
                    patch_size=64,
                    step = 32,
                    output_dim=targetPts,
                    model_dim=64*2,
                    num_heads = 16,
                    num_layers = 3,
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None) 
            elif m == 'TUPE-A':
                from TUPETransformerModel import TUPEOverlappingTransformer
                model = TUPEOverlappingTransformer(
                    context_size=beforePts+afterPts, 
                    patch_size=64,
                    step = 64,
                    output_dim=targetPts,
                    model_dim=64*2,
                    num_heads = 16,
                    num_layers = 3,
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None,
                    only_before=False) 
            elif m == 'TUPE-ALiBi':
                from ALiBiTUPETransformerModel import ALiBiTransformer
                model = ALiBiTransformer(
                    context_size=beforePts+afterPts, 
                    patch_size=64,
                    step = 64,
                    output_dim=targetPts,
                    model_dim=64*2,
                    num_heads = 16,
                    num_layers = 3,
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None,
                    TUPE = True) 
            elif m == 'TUPE-R':
                from RelativeTUPETransformerModel import RelativeTUPETransformer
                model = RelativeTUPETransformer(
                    context_size=beforePts+afterPts, 
                    patch_size=64,
                    step = 64,
                    output_dim=targetPts,
                    model_dim=64*2,
                    num_heads = 16,
                    num_layers = 3,
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None) 
            elif m == 'ALiBi':
                from ALiBiTUPETransformerModel import ALiBiTransformer
                model = ALiBiTransformer(
                    context_size=beforePts+afterPts, 
                    patch_size=64,
                    step = 64,
                    output_dim=targetPts,
                    model_dim=64*2,
                    num_heads = 16,
                    num_layers = 3,
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None,
                    TUPE = False) 
            elif m == 'CH-Indp':
                from ChannelIndpTransformerModel import ChiIndTUPEOverlappingTransformer
                model = ChiIndTUPEOverlappingTransformer(
                    context_size=beforePts+afterPts, 
                    patch_size=64,
                    step = 64,
                    output_dim=targetPts,
                    model_dim=model_d[i],
                    num_heads = 16,
                    num_layers = n_layer[i],
                    lr=0.001,
                    warmup=warmup,
                    max_iters=max_iters,
                    dropout=0.2,
                    input_dropout=0.2,
                    mask = None,
                    only_before=False)
            
            model.load_state_dict(torch.load(model_path + n + '.pt'))
            
            if m == 'CH-Indp':
                
                pred_error = []
                iter_dl_test = iter(CH_dl_test)
                for i in range(int(testSize/3333)):
                    if i % 10 == 0: print("Calculating Prediction Error for Batch no: ", i)
                    x, y = next(iter_dl_test)
                    pred = model(x) 
                    B, C, NP = y.shape
                    y = y.reshape(B*C, NP)
                    pred_er = abs(pred-y)
                    pred_error.append(pred_er.detach()) # Add mean predicion over samples 
                
                abs_pred_error = torch.cat(list(map(torch.tensor, pred_error)), dim=0)
                MSE = torch.mean(torch.mean(torch.square(abs_pred_error), dim=0)) # Overall 
                MAE = torch.mean(abs_pred_error, dim=0)
                
                print("THIS IS m", m+n)
                MAE_MSE[m + n] = {'MAE':float((sum(MAE)/len(MAE)).detach().numpy()), 
                                  'MSE': float(MSE.detach().numpy())}
                print("MAE and MSE: ", MAE_MSE[m+n]) 
            
            else: 
                ds_test, dl_test = getData(testSize, path, beforePts, afterPts, targetPts, 
                                  channelIds, sessionIds,CH=False)
                pred_error = []
                iter_dl_test = iter(dl_test)
                for i in range(int(testSize/10000)):
                    if i % 10 == 0: print("Calculating Prediction Error for Batch no: ", i)
                    x, y = next(iter_dl_test)
                    pred = model(x) 
                    pred_er = abs(pred-y)
                    pred_error.append(pred_er.detach()) # Add mean predicion over samples 
                
                abs_pred_error = torch.cat(list(map(torch.tensor, pred_error)), dim=0)
                
                MSE = torch.mean(torch.mean(torch.square(abs_pred_error), dim=0)) # Overall 
                MAE = torch.mean(abs_pred_error, dim=0)
                
                print("THIS IS", m+n)
                MAE_MSE[m+n] = {'MAE':float((sum(MAE)/len(MAE)).detach().numpy()), 
                                  'MSE': float(MSE.detach().numpy())}
                print("MAE and MSE: ", MAE_MSE[m+n])
        
        MAE_MSE_ALL[r] = MAE_MSE 
    with open('./test_plots/MAE_MSE_model_sizes_all_big.json', 'w') as fp:
        json.dump(MAE_MSE_ALL, fp)
        

if __name__ == '__main__':
    # path= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
    path = '/data/'
    beforePts=512
    afterPts=512
    targetPts=96
    models_to_run = ['CH-Indp']*5
    neptune_names = ['THES-94', 'THES-95', 'THES-96', 'THES-97', 'THES-98']
    
    model_d = [16, 64, 64*2, 64*3, 64*4]
    n_layer = [1, 2, 3, 3*2, 3*5]
    getTestResults(model_d, n_layer, beforePts, afterPts, targetPts, 
                   models_to_run, neptune_names)
