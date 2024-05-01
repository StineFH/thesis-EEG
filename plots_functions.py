import matplotlib.pyplot as plt
import torch


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

def val_train_loss(val_loss, train_loss, x_axis = 'Epochs'):
    assert len(val_loss) == len(train_loss)
    epochs = range(1, len(val_loss)+1)
    
    plt.plot(epochs, val_loss, label='Validation', color = 'b')
    plt.plot(epochs, train_loss, label='Train', color = 'g')
    plt.title('Loss over  ' + x_axis)
    plt.xlabel(x_axis)
    plt.xticks(epochs)
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

def visualizeTargetPrediction(x, y, model, model_path, path, subjectId, sessionId, beforePts, afterPts,
                              targetPts, file_name, only_before):

    # Load saved weights 
    model.load_state_dict(torch.load(model_path))
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



############################ Plot Prediction Error ############################
path= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
subjectId = '001'
sessionId='001'
beforePts=512
afterPts=512
targetPts=96
channelIds=[1,19,23]
# model_path = './linear_model_snapshot/THES-48.pt'
model_path = './transformer_model_snapshot/THES-66.pt'

from LinearModel import linearModel
model = linearModel(lr=0.001,input_size=512+512, output_size=96, 
                        warmup=620,
                        max_iters=18800) 

# from BasicTransformerModel import Transformer
# model = Transformer( # MSE LOSS
#     context_size=beforePts+afterPts, 
#     context_block=50,
#     output_dim=targetPts,
#     model_dim=50,
#     num_heads = 1,
#     num_layers = 1,
#     lr=0.001,
#     warmup=300,
#     max_iters=9200,
#     dropout=0.2,
#     input_dropout=0.2,
#     mask = None) 

# from L1LossTransformerModel import Transformer
# model = Transformer(
#     context_size=beforePts+afterPts, 
#     context_block=50,
#     output_dim=targetPts,
#     model_dim=50,
#     num_heads = 1,
#     num_layers = 1,
#     lr=0.001,
#     warmup=620,
#     max_iters=18800,
#     dropout=0.2,
#     input_dropout=0.2,
#     mask = None)

# from BatchNormTransformerModel import BatchNormTransformer
# model = BatchNormTransformer(
#         context_size=500+500, 
#         context_block=50,
#         output_dim=100,
#         model_dim=50,
#         num_heads=10,
#         num_layers=3,
#         lr=0.001,
#         warmup=620, #620
#         max_iters=18800, #18800
#         dropout=0.2,
#         input_dropout=0.2,
#         mask = None)  

# from TUPETransformerModel import TUPEOverlappingTransformer
# model = TUPEOverlappingTransformer(
#         context_size=512+512,
#         patch_size=32,
#         step = 16,
#         output_dim=96,
#         model_dim=64,
#         num_heads=16,
#         num_layers=3,
#         lr=0.001,
#         warmup=620, #620
#         max_iters=18800, #18800
#         dropout=0.2,
#         input_dropout=0.2,
#         mask = None,
#         only_before=False) 

# from overlappingPatchesTransformer import OverlappingTransformer
# from LogCoshLossTransformerModel import Transformer

# from RelativeTUPETransformerModel import RelativeTUPETransformer
# model = RelativeTUPETransformer(
#     context_size=beforePts+afterPts, 
#     patch_size=64,
#     step = 32,
#     output_dim=targetPts,
#     model_dim=64*2,
#     num_heads = 16,
#     num_layers = 3,
#     lr=0.001,
#     warmup=620,
#     max_iters=18800,
#     dropout=0.2,
#     input_dropout=0.2,
#     mask = None) 

import data_utils4 as du
# import data_utils_channelIndp as du
# from ChannelIndpTransformerModel import ChiIndTUPEOverlappingTransformer

# model = ChiIndTUPEOverlappingTransformer(
#     context_size=beforePts+afterPts, 
#     patch_size=64,
#     step = 32,
#     output_dim=targetPts,
#     model_dim=64,
#     num_heads = 16,
#     num_layers = 3,
#     lr=0.001,
#     warmup=620,
#     max_iters=18800,
#     dropout=0.2,
#     input_dropout=0.2,
#     mask = None,
#     only_before=False) 

# from ALiBiTUPETransformerModel import ALiBiTransformer
# model = ALiBiTransformer(
#     context_size=beforePts+afterPts, 
#     patch_size=64,
#     step = 32,
#     output_dim=targetPts,
#     model_dim=64*2,
#     num_heads = 16,
#     num_layers = 3,
#     lr=0.001,
#     warmup=620,
#     max_iters=18800,
#     dropout=0.2,
#     input_dropout=0.2,
#     mask = None,
#     TUPE = False) 

## GET THE DATA 
def mytransform(raw):
    raw.filter(0.1, 40)
    raw._data=raw._data*1e6
    return raw

subPath=du.returnFilePaths(path, [subjectId], sessionIds=[sessionId])
ds_val=du.EEG_dataset_from_paths(subPath, beforePts=beforePts,afterPts=afterPts,
                                 targetPts=targetPts, channelIdxs=1,
                                 preprocess=False,limit=20,transform=mytransform
                                 )
dl_val=torch.utils.data.DataLoader(ds_val, batch_size=1, shuffle=False)

no = -1
data_iter = iter(dl_val)

x, y = next(data_iter)
no += 1
file_name = './plots/target-pred-THES66-sub001_' + str(no)
visualizeTargetPrediction(x, y, model, model_path, path, subjectId, sessionId,
                          beforePts, afterPts, targetPts,  
                          file_name = file_name,
                          only_before = False)


####################### Plot Absolute Prediction Error ########################

filename = 'THES-75' 
abs_pred_error = torch.load("./lin_model_prediction_error/"  + filename + '.pt')
# abs_pred_error = torch.load("./transformer_prediction_error/"  + filename + '.pt')

abs_prediction_error(abs_pred_error, 
                     file_name='./plots/avg_abs_pred_error_'+filename+'.png',
                     only_before=False)


"""
Validation errors: 
    Vanilla Transformer:
        Avg L1:  tensor(6.4576)
        Avg MSE:  tensor(80.8829)
    Linear: 
        Avg L1:  tensor(6.4724)
        Avg MSE:  tensor(84.5998)
"""