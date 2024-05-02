import matplotlib.pyplot as plt
import torch
import data_utils4 as du
import data_utils_channelIndp as CHdu
import mne_bids as mb

filenames = ['THES-71',
             'THES-72',
             'THES-73',
             'THES-74',
             'THES-75'
             'THES-76',
             'THES-77',
             'THES-78',
             'THES-83']
for i in filenames: 
    abs_pred_error = torch.load("./transformer_prediction_error/"  + i + '.pt')
    MAE = torch.mean(abs_pred_error, dim=0)
    torch.save(MAE, './transformer_prediction_error/' + 'MAE-' + str(i) + '.pt')

beforePts=512
afterPts=512
targetPts=96
channelIds=[1, 19, 23]
sessionIds=['001', '002', '003', '004']

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
    elif m == 'CH-Inp':
        from ChannelIndpTransformerModel import ChiIndTUPEOverlappingTransformer
        model = ChiIndTUPEOverlappingTransformer(
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
        
    model.load_state_dict(torch.load(model_path + n + '.pt'))
    return model

def getData(path,  beforePts, afterPts, targetPts, channelIds, sessionIds,
            CH = False):
    def mytransform(raw):
        raw.filter(0.1, 40)
        raw._data=raw._data*1e6
        return raw
    
    subjectIds=mb.get_entity_vals(path,'subject', with_key=False)
    testIds=subjectIds.copy()[:3]
    
    if CH: 
        subPath = CHdu.returnFilePaths(path, testIds, sessionIds=sessionIds)
        ds_test = CHdu.EEG_dataset_from_paths(subPath, 
                                            beforePts=beforePts,afterPts=afterPts, 
                                            targetPts=targetPts, channelIdxs=channelIds,
                                            preprocess=False,limit=20,
                                            transform=mytransform
                                            )
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False)
    else: 
        subPath = du.returnFilePaths(path, testIds, sessionIds=sessionIds)
        ds_test = du.EEG_dataset_from_paths(subPath, 
                                            beforePts=beforePts,afterPts=afterPts, 
                                            targetPts=targetPts, channelIdxs=channelIds,
                                            preprocess=False,limit=20,
                                            transform=mytransform
                                            )
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False)
    return dl_test

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

def create_prediction_plots(file_name, models, n):

    beforePts = 512
    afterPts = 512
    targetPts = 96
    
    data_iter = iter(dl_test_one)
    
    for i in range(5):  
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
        
        
if __name__ == '__main__':

    path= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
    # path = '/data/'
    save = 'Final_plots_MAE'
    filenames = ['THES-71',
                 'THES-72',
                 'THES-73',
                 'THES-74',
                 'THES-75'
                 'THES-76',
                 'THES-77',
                 'THES-78',
                 'THES-83']
    labels = ['Linear',
              'MSE',
              'L1',
              'LogCosh',
              'Overlapping',
              'TUPE-A',
              'TUPE-ALiBi',
              'TUPE-R',
              'ALiBi',
              'Ch-Indp']
    
    # Get MAE plot over points for all the models in one plot 
    MAE_grouped_plot(filenames, labels, './test_plots/MAE_All')
    
    
    # Get data
    dl_test_one = getData(path,
                          512, 512, 96, [1, 19, 23], ['001', '002', '003', '004'],
                          CH = False)
    CH_dl_test_one = getData(path,
                          512, 512, 96, [1, 19, 23], ['001', '002', '003', '004'],
                          CH = True)
    
    file_name=''
    models = ['CH-Indp',
              'linear_model',
              'vanilla',
              'L1',
              'LogCosh',
              'overlapping',
              'TUPE-A',
              'TUPE-ALiBi',
              'TUPE-R',
              'ALiBi'
              ]
    n = ['THES-83',
         'THES-71', 
         'THES-70',
         'THES-72',
         'THES-73',
         'THES-74',
         'THES-75',
         'THES-76',
         'THES-77',
         'THES-78'
         ]
    
    create_prediction_plots(file_name, models, n)