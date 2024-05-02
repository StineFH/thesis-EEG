import matplotlib.pyplot as plt
import torch
import data_utils4 as du
import data_utils_channelIndp as CHdu
import mne_bids as mb

# abs_pred_error = torch.load("./transformer_prediction_error/"  + 'THES-70' + '.pt')

# abs_pred_error = torch.cat(list(map(torch.tensor, abs_pred_error)), dim=0)
# MSE = torch.mean(torch.mean(torch.square(abs_pred_error), dim=0)) # Overall 
# MAE = torch.mean(abs_pred_error, dim=0)

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
    colors = {0: '#4477AA', 2: '#228833', 7:'#CCBB44', 9: '#AA3377', 
              'others': '#BBBBBB'}
    #https://personal.sron.nl/~pault/#sec:qualitative
    ax = plt.axes()
    ax.set_facecolor("#F8F8F8")
    
    for i in range(len(filenames)):
        MAE = torch.load("./transformer_prediction_error/" + 'MAE-' + filenames[i] + '.pt')
        dist_from_known = range(1, len(MAE)+1) 
        if i in [0, 2, 7, 9]:
            plt.plot(dist_from_known, MAE, label=labels[i], color = colors[i])
        else: 
            plt.plot(dist_from_known, MAE, label=labels[i], 
                     color = colors['others'], alpha=0.4)
    plt.title('')
    plt.xlabel('Distance from last point before target')
    plt.xticks(list(dist_from_known[::10]))
    plt.ylabel('Average absolute prediction error')
    
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #             ncol=5, fancybox=True, shadow=False)
    plt.legend(loc='center right', bbox_to_anchor=(1.16, 0.5),
                ncol=1, fancybox=True, shadow=False)
    if save:
        figure = plt.gcf()
        figure.set_size_inches(12, 8)
        plt.savefig(save, dpi = 100)
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


def modelSizesPlot(layers, file_name, save=None):
    MAE_MSE_sizes = torch.load('./test_plots/' + file_name + '.pt')
    MAE_MSE_sizes = {'16':{'MAE':0.4, 'MSE':3}, '64':{'MAE':0.2, 'MSE':6}, 
                     '128':{'MAE':0.4, 'MSE':5}}       
    
    x_axis = [d+"/"+ str(l) for d, l in zip(MAE_MSE_sizes.keys(), layers)]
    MAE, MSE = list(zip(*list(map(lambda x: x.values(), MAE_MSE_sizes.values()))))
    
    plt.plot(x_axis, MAE)
    plt.plot(x_axis, MAE, marker = 'o')
    plt.title('Model size')
    plt.xlabel('model_dim/n_layers')
    plt.ylabel('Validation loss')
    if save:
        figure = plt.gcf()
        figure.set_size_inches(12, 8)
        plt.savefig(save, dpi = 100)
    plt.show()


if __name__ == '__main__':

    path= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
    # path = '/data/'
    save = 'Final_plots_MAE'
    filenames = ['THES-71',
                 'THES-70',
                 'THES-72',
                 'THES-73',
                 'THES-74',
                 'THES-75',
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
    
    ######################## MODEL SIZES #########################
    layers = [1, 2, 3, 3*2, 3*5, 3*8, 3*12]
    modelSizesPlot(layers, 'validation_loss_model_sizes')
