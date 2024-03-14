import matplotlib.pyplot as plt
import torch
import data_utils4 as du
from LinearModel import linearModel
from TransformerModel import Transformer



def abs_prediction_error(abs_pred_error, file_name):
    """
    X axis is the distance in points to last know point. 
    The first half and last half of points are displayed where the last half 
    has reversed points to fit with distance to points after. 
    """
    abs_pred_error = list(zip(*abs_pred_error))
    abs_pred_error = list(map(lambda x: sum(x)/10, abs_pred_error))
    
    print("Avg L1: ", sum(abs_pred_error)/len(abs_pred_error))
    
    assert len(abs_pred_error) % 2 == 0
    
    half_len_preds = int(len(abs_pred_error)/2)
    dist_from_known = range(1, half_len_preds+1) 
    
    colors = plt.cm.Paired([1,3])
    ax = plt.axes()
    ax.set_facecolor("#F8F8F8")
    
    plt.plot(dist_from_known, abs_pred_error[:half_len_preds], label='First', color = colors[0])
    plt.plot(dist_from_known, list(abs_pred_error[half_len_preds:])[::-1], label='Last', color = colors[1])
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

def visualizeTargetPrediction(model, model_path, path, subjectId, sessionId, beforePts, afterPts,
                              targetPts, channelIds, file_name):
    # Get data 
    
    def mytransform(raw):
        raw.filter(0.1, 40)
        raw._data=raw._data*1e6
        return raw
    
    subPath=du.returnFilePaths(path, [subjectId], sessionIds=[sessionId])[0]
    
    ds_train=du.EEG_dataset_from_paths([subPath], beforePts=beforePts,
                                       afterPts=afterPts,targetPts=targetPts, 
                                       channelIdxs=channelIds, preprocess=False,
                                       limit=1, transform=mytransform)
    dl_train=torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=False)

    x, y = next(iter(dl_train))

    # Load saved weights 
    model.load_state_dict(torch.load(model_path))
    pred = model(x) 
    
    x1, x2= x # x1 before and x2 after window
    original  = torch.cat((x1, y, x2),dim=1)
    
    # Plotting
    colors = plt.cm.Paired([1,5])
    ax = plt.axes()
    ax.set_facecolor("#F8F8F8")
    
    ax.axvline(beforePts, color = "grey", linestyle = 'dashed')
    ax.axvline(beforePts+targetPts, color = "grey", linestyle = 'dashed')
    
    plt.plot(range(beforePts+targetPts+afterPts), original[0].detach().numpy(), 
             label='Original', color = colors[0])
    plt.plot(range(beforePts, beforePts+targetPts), pred[0].detach().numpy(), 
             label='Prediction', color = colors[1])
    
    plt.title('Predicted and target EEG')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()

    if file_name: 
        plt.savefig(file_name)
    plt.show()



############################ Plot Prediction Error ############################
path= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
subjectId = '001'
sessionId='001'
beforePts=500
afterPts=500
targetPts=100
channelIds=[1,19,23]
# model_path = './linear_model_snapshot/THES-30.pt'
model_path = './transformer_model_snapshot/THES-32.pt'

# model = linearModel(0.001, beforePts+afterPts, targetPts)
model = Transformer(
        context_size=500+500, 
        context_block=50,
        output_dim=100,
        model_dim=50,
        num_heads=1,
        num_layers=1,
        lr=0.001,
        warmup=300,
        max_iters=3300,
        dropout=0.0,
        input_dropout=0.0,
        mask = None) 
file_name = './plots/target-pred-THES32-sub001'

visualizeTargetPrediction(model, model_path, path, subjectId, sessionId, 
                          beforePts, afterPts, targetPts, channelIds, 
                          file_name = file_name)

####################### Plot Absolute Prediction Error ########################


filename = 'THES-30'
abs_pred_error = torch.load("./lin_model_prediction_error/"  + filename+ '.pt')
# abs_pred_error = torch.load("./transformer_prediction_error/"  + filename + '.pt')

abs_prediction_error(abs_pred_error, file_name='./plots/avg_abs_pred_error_'+filename+'.png')


