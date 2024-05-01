import matplotlib.pyplot as plt
import torch
import data_utils4 as du
# import data_utils_channelIndp as CHdu
import mne_bids as mb
# import json
# During test_data_results.py save the pred_error over dist 

def get_model(m, n, beforePts, afterPts, targetPts):
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
    
    model.load_state_dict(torch.load(model_path + n + '.pt'))
    return model

save = 'Final_plots_MAE'
filenames = ['THES-75',
             'THES-76']
labels = ['L1',
          'TUPE']

def MAE_grouped_plot(filenames, labels, save):
    colors = plt.cm.tab10([1,len(filenames)])
    ax = plt.axes()
    ax.set_facecolor("#F8F8F8")
    
    for i in range(len(filenames)):
        MAE = torch.load("./transformer_prediction_error/"  + filenames[i] + '.pt')
        dist_from_known = range(1, MAE+1) 
        assert len(MAE) % 2 == 0
        plt.plot(dist_from_known, MAE, label=labels[i], color = colors[i])
    plt.title('Average absolute prediction error')
    plt.xlabel('Distance from last point before target')
    plt.xticks(list(dist_from_known[::4]))
    plt.ylabel('Average absolute prediction error')
    plt.legend()
    if save:
        plt.savefig(save)
    plt.show()
    
    
def getData(testSize, path,
            beforePts, afterPts, targetPts, channelIds, sessionIds):
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
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False,
                                          num_workers = 8)
    return dl_test


dl_test_one = getData()

file_name=''
models = []
n = []
beforePts = 512
afterPts = 512
targetPts = 96


data_iter = iter(dl_test_one)

for i in range(5): # Number of plots 
    x, y = next(data_iter)
    
    colors = plt.cm.Paired([1,5])
    ax = plt.axes()
    ax.set_facecolor("#F8F8F8")
    for i in range(models):
        model = get_model(models[i], n[i], beforePts, afterPts, targetPts)
        pred = model(x) 
    
        
        x1, x2= x # x1 before and x2 after window
        plt.plot(range(1, pred+1), pred[0].detach().numpy(), 
                  label=models[i], color = colors[0])
    plt.plot(range(1, targetPts+1), y[0].detach().numpy(), 
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