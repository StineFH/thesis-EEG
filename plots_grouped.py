import matplotlib.pyplot as plt
import torch
import data_utils4 as du
import data_utils_channelIndp as CHdu
import mne_bids as mb
import json

from fig_size_func import set_size

import pandas as pd
# abs_pred_error = torch.load("./transformer_prediction_error/"  + 'THES-90' + '.pt')

# abs_pred_error = torch.cat(list(map(torch.tensor, abs_pred_error)), dim=0)
# MSE = torch.mean(torch.mean(torch.square(abs_pred_error), dim=0)) # Overall 
# MAE = torch.mean(abs_pred_error, dim=0)

#Colors: #https://personal.sron.nl/~pault/#sec:qualitative

width = 426.79134

beforePts=512
afterPts=512
targetPts=96
channelIds=[1, 19, 23]
sessionIds=['001', '002', '003', '004']
width = 345

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern Serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}


def get_model(m, n, beforePts, afterPts, targetPts):
    if m == "Linear":
        model_path = './linear_model_snapshot/'
    else:
        model_path = './transformer_model_snapshot/'
    
    warmup=6250
    max_iters=188000
    
    if m == "Linear":
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
                                            preprocess=False,limit=30,
                                            transform=mytransform
                                            )
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False)
    else: 
        subPath = du.returnFilePaths(path, testIds, sessionIds=sessionIds)
        ds_test = du.EEG_dataset_from_paths(subPath, 
                                            beforePts=beforePts,afterPts=afterPts, 
                                            targetPts=targetPts, channelIdxs=channelIds,
                                            preprocess=False,limit=30,
                                            transform=mytransform
                                            )
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False)
    return dl_test

def MAE_grouped_plot(filenames, labels, save):
    colors = {0: '#004488', 9: '#BB5566', 'others': '#BBBBBB'}
    
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=0.85))
    plt.rcParams.update(tex_fonts)
    for i in range(len(filenames)):
        MAE = torch.load("./transformer_prediction_error/" + 'MAE-' + filenames[i] + '.pt')
        dist_from_known = range(1, len(MAE)+1) 
        if i in [0, 9]:
            ax.plot(dist_from_known, MAE, label=labels[i], color = colors[i])
        else: 
            ax.plot(dist_from_known, MAE, label=labels[i], 
                     color = colors['others'], alpha=0.4)
    plt.title('')
    plt.xlabel('Distance from last \n point before target')
    plt.xticks(list(dist_from_known[::10]))
    plt.ylabel('Test MAE')
    
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.84),
    #             ncol=2, fancybox=True, shadow=False)
    plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5),
                ncol=1, fancybox=True, shadow=False)
    # plt.legend(loc='center right', bbox_to_anchor=(2.24, 0.5),
    #             ncol=2, fancybox=True, shadow=False)
    if save:
        fig.savefig('./test_plots/' + save + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
def VaryingContextSizePlot(labels, save):
    f = open('./test_plots/MAE_MSE_test_varying.json')
    MAE = json.load(f)
    f.close()
    MAE_error = list(zip(*list(map(lambda x: x.values(), MAE.values()))))
    colors = {0: '#004488', 1: '#BB5566'}
    x = ['128', '256', '512', '1024', '2048']

    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=0.60))
    plt.rcParams.update(tex_fonts)
    plt.plot(x, MAE_error[0][0:5], label=labels[0], color = colors[0])
    plt.plot(x, MAE_error[0][0:5], marker = 'o', color = colors[0])
    plt.plot(x, MAE_error[0][5:], label=labels[1], color = colors[1])
    plt.plot(x, MAE_error[0][5:], marker = 'o', color = colors[1])
    # plt.title('Mean absolute prediction error')
    plt.title('Varying context size')
    plt.xlabel('Size of input before and after target')
    plt.xticks(x)
    plt.ylabel('Average test MAE')
    plt.legend()
    if save:
        fig.savefig('./test_plots/' + save + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def modelSizesPlot(S_filename, B_filename, labels, save_MAE=None, save_MSE=None):
    layers = [1, 2, 3, 3*2, 3*5, 3*8]
    model_dim = [16, 64, 64*2, 64*3, 64*4]
    f = open('./test_plots/' + S_filename + '.json')
    S_MAE = json.load(f)
    f.close()
    
    f = open('./test_plots/' + B_filename + '.json')
    B_MAE = json.load(f)
    f.close()
    
    # Structure the data 
    df_S = pd.DataFrame(columns = ['round', 'Model', 'MAE', 'MSE'])
    df_B = pd.DataFrame(columns = ['round', 'Model', 'MAE', 'MSE'])
    for r in range(10):
        df_temp= {}
        dicts = [S_MAE[str(r)][m].values() for m in S_MAE[str(r)]]
        df_temp['MAE'], df_temp['MSE'] = list(zip(*dicts))
        df_temp['Model'] = S_MAE[str(r)].keys()
        df_temp['round'] =  [str(r)]*5
    
        df_S = pd.concat([df_S, pd.DataFrame.from_dict(df_temp)])
        
        df_temp_B= {}
        dicts = [B_MAE[str(r)][m].values() for m in B_MAE[str(r)]]
        df_temp_B['MAE'], df_temp_B['MSE'] = list(zip(*dicts))
        df_temp_B['Model'] = B_MAE[str(r)].keys()
        df_temp_B['round'] =  [str(r)]*5
    
        df_B = pd.concat([df_B, pd.DataFrame.from_dict(df_temp_B)])
    
    MAE_B = df_B.groupby('Model', as_index=False).agg(
                      {'MAE':['mean','std'], 'MSE':['mean','std']})
    MAE_S = df_S.groupby('Model', as_index=False).agg(
                      {'MAE':['mean','std'], 'MSE':['mean','std']})
    
    # Make the MAE plot
    colors = {0: '#004488', 1: '#BB5566'}
    
    fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=0.6))
    plt.rcParams.update(tex_fonts)
    x_axis = [str(d)+"/"+ str(l) for d, l in zip(model_dim, layers)]
    plt.plot(x_axis, MAE_S['MAE']['mean'], color = colors[0], label=labels[0])
    plt.plot(x_axis, MAE_S['MAE']['mean'], color = colors[0],marker ='o')
    # plt.errorbar(x_axis, MAE_S['MAE']['mean'], yerr=MAE_S['MAE']['std'],
    #              linestyle='None', fmt="o", color = colors[0], alpha=0.6)
    
    plt.plot(x_axis, MAE_B['MAE']['mean'], color = colors[1], label=labels[1])
    plt.plot(x_axis, MAE_B['MAE']['mean'], color = colors[1],marker ='o')
    # plt.errorbar(x_axis, MAE_B['MAE']['mean'], yerr=MAE_B['MAE']['std'], 
    #              linestyle='None', fmt="o", color = colors[1], alpha=0.6)

    plt.title('Model size')
    plt.xlabel('model_dim/n_layers')
    plt.ylabel('Average test MAE')
    plt.legend(title = 'Dataset Size')
    if save_MAE:
        fig.savefig('./test_plots/' + save_MAE + '.pdf', 
                    format='pdf', bbox_inches='tight')
    plt.show()


def createAllPredictionPlots(file_name, models, n, CH_dl_test_one):
    beforePts = 512
    afterPts = 512
    targetPts = 96
    
    data_iter = iter(CH_dl_test_one)
    for i in range(30): # Number of different spots  
        
        colors = {0: '#DDAA33', 1: '#BB5566', 2:'#004488'}

        x, y = next(data_iter)
        
        for c in range(3): # Making a plot for each channel 
            plt.rcParams.update(tex_fonts)
            fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=0.60))
        
            for j in range(len(models)):
                print(models[j])
                if models[j] == 'CH-Indp':
                    model = get_model(models[j], n[j], beforePts, afterPts, targetPts)
                    pred = model(x) 
                    
                    plt.plot(range(1, targetPts+1), pred[c,:].detach().numpy(),
                             color = colors[j+1],label=models[j])
                else: 
                    x1, x2 = x
    
                    model = get_model(models[j], n[j], beforePts, afterPts, targetPts)
                    pred = model([x1[:,c,:], x2[:,c,:]]) 
                    
                    plt.plot(range(1, targetPts+1), pred[0].detach().numpy(), 
                             color = colors[j+1], label=models[j])
                
            plt.plot(range(1, targetPts+1), y[0,c,:].detach().numpy(), 
                      label='Target', color = colors[0])
            plt.title('Predicted and target EEG')
            plt.xlabel('Time [samples]')
            plt.ylabel(r'Signal [$\mu V$]')
            # plt.legend()
            if file_name: 
                fig.savefig('./test_plots/' + file_name + str(i) + '_' +str(c) + '.png', 
                            format='png', bbox_inches='tight')
            plt.show()

def createThePredictionPlots(file_name, models, n, CH_dl_test_one):
    beforePts = 512
    afterPts = 512
    targetPts = 96
    data_iter = iter(CH_dl_test_one)  
    
    colors = {0: '#DDAA33', 1: '#BB5566', 2:'#004488'}
    c = [2, 2, 1, 0]
    plot_num = -1
    fig, axs = plt.subplots(2, 2, sharex =True,constrained_layout = True,
                            figsize=set_size(width, fraction=1.15, subplots=(2, 2)))
    
    plt.rcParams.update(tex_fonts)    
    for i in range(30): # Number of different data points
        x, y = next(data_iter)      
        if i in [2, 8, 9, 27]:
            plot_num += 1
            for j in range(len(models)):
                if models[j] == 'CH-Indp':
                    model = get_model(models[j], n[j], beforePts, afterPts, targetPts)
                    pred = model(x) 
                    
                    axs.flat[plot_num].plot(range(1, targetPts+1), pred[c[plot_num],:].detach().numpy(),
                             color = colors[j+1],label=models[j])
                else: 
                    x1, x2 = x
    
                    model = get_model(models[j], n[j], beforePts, afterPts, targetPts)
                    pred = model([x1[:,c[plot_num],:], x2[:,c[plot_num],:]]) 
                    
                    axs.flat[plot_num].plot(range(1, targetPts+1), pred[0].detach().numpy(), 
                             color = colors[j+1], label=models[j])
            print("Plot", plot_num, str(i) + '_' + str(c[plot_num]))
            axs.flat[plot_num].plot(range(1, targetPts+1), y[0,c[plot_num],:].detach().numpy(), 
                          label='Target', color = colors[0])
    plt.setp(axs.flat[[1, 3]], xticks=range(1, targetPts+1, 15))
    fig.suptitle('Predicted and target EEG')
    fig.supxlabel('Time [samples]')
    fig.supylabel(r'Signal [$\mu V$]')
    handles, labels = axs.flat[3].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.2, 0.5),
    #             ncol=1, fancybox=True, shadow=False)
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.06, 0.91),
                ncol=1, fancybox=True, shadow=False)
    # plt.tight_layout()
    fig.savefig('./test_plots/' + 'Prediction_plots2' + '.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

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
    MAE_grouped_plot(filenames, labels, 'MAE_All2')
    
    # ############################# MODEL SIZES ################################
    
    S_filename = 'MAE_MSE_model_sizes_all_small'
    B_filename = 'MAE_MSE_model_sizes_all_big'
    labels = ['2M', '4M']
    modelSizesPlot(S_filename, B_filename, labels, 
                   'MAE_loss_model_sizes',
                   'MSE_loss_model_sizes')
    
    # ########################### Varying Context Size ##########################
    
    labels = ['Linear', 'CH-Indp']
    VaryingContextSizePlot(labels, save = 'val_loss_varying_context')
    
    #####################################################################
    # path= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
    # # path = '/data/'
    
    # # Get data

    # CH_dl_test_one = getData(path,
    #                       512, 512, 96, [1, 19, 23], ['001', '002', '003', '004'],
    #                       CH = True)
    
    # models = ['CH-Indp',
    #           'Linear'
    #           ]
    # n = ['THES-83',
    #       'THES-71'
    #       ]
    
    # createThePredictionPlots('predictions_', models, n, CH_dl_test_one)

    # createAllPredictionPlots('predictions_', models, n, CH_dl_test_one)
