"""
1. Import test data: first three subjects 
2. Calculate average L1 and MSE
3. Calculate avg L1 and MSE for each channel? 
4. Visualize 5 pictures of prediction and original signal for each channels  
"""

import matplotlib.pyplot as plt
import torch
import data_utils4 as du
import data_utils_channelIndp as CHdu
import mne_bids as mb
import json

def abs_prediction_error(abs_pred_error, file_name, only_before):
    """
    X axis is the distance in points to last know point. 
    The first half and last half of points are displayed where the last half 
    has reversed points to fit with distance to points after. 
    """
    abs_pred_error = torch.cat(list(map(torch.tensor, abs_pred_error)), dim=0)
    MSE = torch.mean(torch.mean(torch.square(abs_pred_error), dim=0)) # Overall 
    
    MAE = torch.mean(abs_pred_error, dim=0)
    print("Avg L1: ", sum(MAE)/len(MAE))
    print("Avg MSE: ", MSE)
    
    if only_before: 
        len_MAE = len(MAE)
        dist_from_known = range(1, len_MAE+1) 
        
        colors = plt.cm.Paired([1,3])
        ax = plt.axes()
        ax.set_facecolor("#F8F8F8")
        
        plt.plot(dist_from_known, MAE, label='First', color = colors[0])
        plt.title('Average absolute prediction error')
        plt.xlabel('Distance from known point')
        plt.xticks(list(dist_from_known[::4]))
        plt.ylabel('Average absolute prediction error')
        if file_name:
            plt.savefig(file_name)
        plt.show()
        
    else: 
        assert len(MAE) % 2 == 0
        
        half_len_preds = int(len(MAE)/2)
        dist_from_known = range(1, half_len_preds+1) 
        
        colors = plt.cm.Paired([1,3])
        ax = plt.axes()
        ax.set_facecolor("#F8F8F8")
        
        plt.plot(dist_from_known, MAE[:half_len_preds], label='First', color = colors[0])
        plt.plot(dist_from_known, list(MAE[half_len_preds:])[::-1], label='Last', color = colors[1])
        plt.title('Average absolute prediction error')
        plt.xlabel('Distance from known point')
        plt.xticks(list(dist_from_known[::4]))
        plt.ylabel('Average absolute prediction error')
        plt.legend()
        if file_name:
            plt.savefig(file_name)
        plt.show()
    return {'MAE': float((sum(MAE)/len(MAE)).detach().numpy()), 'MSE': float(MSE.detach().numpy())}    
    
def visualizeTargetPrediction(x, y, model, file_name, only_before):

    # Load saved weights 
    pred = model(x) 
    
    if only_before:
        original  = torch.cat((x[:,-100:], y),dim=1)
        
        # Plotting
        colors = plt.cm.Paired([1,5])
        ax = plt.axes()
        ax.set_facecolor("#F8F8F8")
        
        ax.axvline(100, color = "grey", linestyle = 'dashed')
        
        plt.plot(range(100+targetPts), original[0].detach().numpy(), 
                  label='Original', color = colors[0])
        plt.plot(range(100, 100+targetPts), pred[0].detach().numpy(), 
                  label='Prediction', color = colors[1])
        plt.title('Predicted and target EEG')
        plt.xlabel('')
        plt.ylabel('')
        plt.legend()
    
        if file_name: 
            figure = plt.gcf()
            figure.set_size_inches(12, 8)
            plt.savefig(file_name, dpi = 100, bbox_inches='tight')
        plt.show()
    else: 
        x1, x2= x # x1 before and x2 after window
        original  = torch.cat((x1[:,-50:], y, x2[:,:50]),dim=1)
        
        # Plotting
        colors = plt.cm.Paired([1,5])
        ax = plt.axes()
        ax.set_facecolor("#F8F8F8")
        
        ax.axvline(50, color = "grey", linestyle = 'dashed')
        ax.axvline(50+targetPts, color = "grey", linestyle = 'dashed')
        
        plt.plot(range(50+targetPts+50), original[0].detach().numpy(), 
                  label='Original', color = colors[0])
        plt.plot(range(50, 50+targetPts), pred[0].detach().numpy(), 
                  label='Prediction', color = colors[1])
        
        plt.title('Predicted and target EEG')
        plt.xlabel('')
        plt.ylabel('')
        plt.legend()
    
        if file_name: 
            figure = plt.gcf()
            figure.set_size_inches(12, 8)
            plt.savefig(file_name, dpi = 100, bbox_inches='tight')
        plt.show()

# Only channel independent needs it's own data 

############################### Import test data ##############################

def getData(testSize, path, beforePts, afterPts, targetPts, channelIds, sessionIds):
    def mytransform(raw):
        raw.filter(0.1, 40)
        raw._data=raw._data*1e6
        return raw
    
    subjectIds=mb.get_entity_vals(path,'subject', with_key=False)
    testIds=subjectIds.copy()[:3]
    subPath = du.returnFilePaths(path, testIds, sessionIds=sessionIds)
    ds_test = du.EEG_dataset_from_paths(subPath, 
                                        beforePts=beforePts,afterPts=afterPts, 
                                        targetPts=targetPts, channelIdxs=channelIds,
                                        preprocess=False,limit=testSize,
                                        transform=mytransform
                                        )
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=10000, shuffle=False,
                                          num_workers = 8)
    return ds_test, dl_test
# Import data channel independent 

############################### Get MSE and MAE ###############################


def getTestResults(models_to_run, neptune_names, ds_test, dl_test):
    dl_test_one = torch.utils.data.DataLoader(ds_test, batch_size=1, 
                                              shuffle=False, num_workers = 8)
    MAE_MSE = {}
    for m, n in zip(models_to_run, neptune_names):
        if m == "linear_model":
            model_path = './linear_model_snapshot/'
        else:
            model_path = './transformer_model_snapshot/'
        plot_destination = './test_plots/'
        
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
        
        
        model.load_state_dict(torch.load(model_path + n + '.pt'))
        pred_error = []
        iter_dl_test = iter(dl_test)
        for i in range(int(testSize/10000)):
            if i % 10 == 0: print("Calculating Prediction Error for Batch no: ", i)
            x, y = next(iter_dl_test)
            pred = model(x) 
            pred_er = abs(pred-y)
            pred_error.append(pred_er.detach()) # Add mean predicion over samples 
        
        print("THIS IS m", m)
        out = abs_prediction_error(pred_error, 'plot_destination' + n, 
                                   only_before = False)
        print(out)
        MAE_MSE[m] = out
        print("MAE and MSE: ", MAE_MSE[m])
        # Also different for channel indp. 
        
        ####################### Visualize Prediction + original #######################
        # Could include a picture that 
        
        print("Now making plots for prediction")
        no = -1
        data_iter = iter(dl_test_one)
        for i in range(5):
            x, y = next(data_iter)
            no += 1
            file_name = plot_destination +'target-pred-' + n + str(no)
            visualizeTargetPrediction(x, y, model, file_name, only_before=False)
    

    with open('./test_plots/MAE_MSE.json', 'w') as fp:
        json.dump(MAE_MSE, fp)
        

if __name__ == '__main__':
    # path= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
    path = '/data/'
    beforePts=512
    afterPts=512
    targetPts=96
    channelIds=[1,19,23]
    testSize = 1875000
    sessionIds = ['001', '002', '003', '004']
    
    ds_test, dl_test = getData(testSize, path, beforePts, afterPts, targetPts, 
                      channelIds, sessionIds)
    models_to_run = [#'linear_model', 
                     #'vanilla',
                     'L1',
                     #'LogCosh',
                     #'overlapping',
                     'TUPE-A',
                     'TUPE-ALiBi',
                     'TUPE-R',
                     'ALiBi'
                     ]
    neptune_names = [#'THES-71', 
                     #'THES-70',
                     'THES-72',
                     #'THES-73',
                     #'THES-74',
                     'THES-75',
                     'THES-76',
                     'THES-77',
                     'THES-78'
                     ]
    getTestResults(models_to_run, neptune_names, ds_test, dl_test)

"""Test results
    ##
Linear model:
    Avg L1:  tensor(8.2442)
    Avg MSE:  tensor(142.0788)
    
Vanilla Transformer: 
    Avg L1:  tensor(8.3323)
    Avg MSE:  tensor(139.8540)

L1 Transformer: 
    Avg L1:  tensor(8.2164)
    Avg MSE:  tensor(136.5007)

LogCosh Transformer:    
    Avg L1:  tensor(8.2945)
    Avg MSE:  tensor(138.7707)

Overlapping: 
    Avg L1:  tensor(8.2233)
    Avg MSE:  tensor(136.8165)

"""

