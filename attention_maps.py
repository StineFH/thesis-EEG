import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from BasicTransformerModel import Transformer
from plots_grouped import get_model, getData
from fig_size_func import set_size

beforePts=512
afterPts=512
targetPts=96
warmup=6250
max_iters=188000



width = 426.79134
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


def VisualizeAttentionMaps(model_name, nep_name, CH_dl_test_one, 
                           file_name):
    colors = {0: '#DDAA33', 1: '#BB5566', 2:'#004488', 3:'#228833'}
    model = get_model(model_name, nep_name, 512, 512, 96)
    
    data_iter = iter(CH_dl_test_one) 
    
    fig, axs = plt.subplots(6, 2, sharex='col', figsize=set_size(width, fraction=0.75, 
                                                  subplots=(6, 2)), layout ='constrained')
    plt.rcParams.update(tex_fonts)
    axes = axs.ravel()
    cs = [[0, 1, 2], [0, 1, 2]] # Channels
    plot_num = 0
    
    if model_name == 'CH-Indp': 
        for i in range(30):
            inputs, y = next(data_iter)
            
            if i in [15, 28]: # sample
                for c in range(len(cs[[15, 28].index(i)])):
                    x1_, x2_ = inputs
                    
                    #input x1 and x2: returns as (batch,sequence,patch_size)
                    x1=model.contextBlockFunc(x1_)
                    x2=model.contextBlockFunc(x2_)
                    x=torch.cat((x1,x2),dim=2) # second dimension because of channels now in first
                    B, C, NP, LP = x.shape
                    x = x.reshape(B*C, NP, LP)
                    pred = model(inputs)
                    
                    attn_CH = model.get_attention_maps(x, average_attn_weights=True)
                    attn_maps_samples = torch.mean(attn_CH, dim =0) # Mean over layers -> one attention map for each input 
                    
                    if plot_num == 0:
                        im0 = axes[plot_num].imshow(attn_maps_samples[c,:,:].detach().numpy(), 
                                                         norm = None, vmin=0, vmax=0.25)
                    axes[plot_num].imshow(attn_maps_samples[c,:,:].detach().numpy(), 
                                                     norm = None, vmin=0, vmax=0.25)
                    axes[plot_num].set_ylabel(r'Patches')
                    
                    axes[plot_num+1].plot(range(1, 96+101), torch.cat([x1_[0,c,462:],
                                                                      y[0,c,:],x2_[0,c,0:50]]).detach().numpy(), 
                                          color = colors[0], label='Target')
                    axes[plot_num+1].plot(range(51, 96+51), pred[c,:].detach().numpy(), 
                                          color = colors[2], label='Forecast') 
                    axes[plot_num+1].axvline(50, color = 'black', ls = '--', alpha=0.4)
                    axes[plot_num+1].axvline(51+96, color = 'black', ls = '--', alpha=0.4)
                    axes[plot_num+1].set_ylabel(r'Signal [$\mu V$]')
                    plot_num += 2  
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center',
                   ncol=1, fancybox=True, shadow=False, bbox_to_anchor=(0.49, 0.977))
        plt.xticks([])  
        plt.setp(axes[10], xticks=range(0, 16, 3))
        plt.setp(axes[11], xticks=range(1, targetPts+51+50, 30))
        axs[0][0].set_title('Attention Maps')
        axs[0][1].set_title('Targets and Forecasts')
        axs[5, 0].set_xlabel('Patches')
        axs[5, 1].set_xlabel('Time [samples]')
        plt.colorbar(im0, ax=axs[:, [0]], location='right', shrink = 0.35)
        fig.savefig('./test_plots/' + file_name + '.pdf',
                    format='pdf', bbox_inches='tight')  
        plt.show()
    
    
    elif  model_name == 'TUPE-R':
        for i in range(30):
            inputs, y = next(data_iter)
            
            if i in [15, 28]: # sample
                for c in range(len(cs[[15, 28].index(i)])):
                    x1_, x2_ = inputs
                    pos = [15, 28].index(i)
                    x1_ = x1_[:,cs[pos][c], :]
                    x2_ = x2_[:,cs[pos][c], :]
                    x1=model.contextBlockFunc(x1_)
                    x2=model.contextBlockFunc(x2_)
                    x=torch.cat((x1,x2),dim=1)
                    pred = model([x1_, x2_])
                    
                    attn_CH = model.get_attention_maps(x, average_attn_weights=True)
                    attn_maps_samples = torch.mean(attn_CH, dim =0) # Mean over layers -> one attention map for each input 
                    
                    if plot_num == 0:
                        im0 = axes[plot_num].imshow(attn_maps_samples[0,:,:].detach().numpy(), 
                                                         norm = None, vmin=0, vmax=0.25)
                    axes[plot_num].imshow(attn_maps_samples[0,:,:].detach().numpy(), 
                                                     norm = None, vmin=0, vmax=0.25)
                    axes[plot_num].set_ylabel(r'Patches')
                    
                    axes[plot_num+1].plot(range(1, 96+101), torch.cat([x1_[0,462:],
                                                                      y[0,c,:],x2_[0,0:50]]).detach().numpy(), 
                                          color = colors[0], label='Target')
                    axes[plot_num+1].plot(range(51, 96+51), pred[0,:].detach().numpy(), 
                                          color = colors[2], label='Forecast') 
                    axes[plot_num+1].axvline(50, color = 'black', ls = '--', alpha=0.4)
                    axes[plot_num+1].axvline(51+96, color = 'black', ls = '--', alpha=0.4)
                    axes[plot_num+1].set_ylabel(r'Signal [$\mu V$]')
                    plot_num += 2  
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center',
                   ncol=1, fancybox=True, shadow=False, bbox_to_anchor=(0.49, 0.977))
        plt.xticks([])  
        plt.setp(axes[10], xticks=range(0, 16, 3))
        plt.setp(axes[11], xticks=range(1, targetPts+51+50, 30))
        axs[0][0].set_title('Attention Maps')
        axs[0][1].set_title('Targets and Forecasts')
        axs[5, 0].set_xlabel('Patches')
        axs[5, 1].set_xlabel('Time [samples]')
        plt.colorbar(im0, ax=axs[:, [0]], location='right', shrink = 0.35)
        fig.savefig('./test_plots/' + file_name + '.pdf',
                    format='pdf', bbox_inches='tight')  
        plt.show()
                    
        
path= 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
CH_dl_test_one = getData(path, 512, 512, 96, [1, 19, 23], ['001'], CH=True)


nep_name = 'THES-83'
model_name = 'CH-Indp'

VisualizeAttentionMaps(model_name, nep_name,CH_dl_test_one,'attention_maps_CH_Indp2')

model = get_model('Linear', 'THES-71', 512, 512, 96)

nep_name = 'THES-77'
model_name = 'TUPE-R'
VisualizeAttentionMaps(model_name, nep_name,CH_dl_test_one,'attention_maps_TUPER')
        
################################### Testing ###################################

n = 'THES-77'
model = get_model('TUPE-R', n, beforePts, afterPts, targetPts)

data_iter = iter(CH_dl_test_one)

inputs, y = next(data_iter)
x1_, x2_ = inputs
x1=model.contextBlockFunc(x1_)
x2=model.contextBlockFunc(x2_)
x=torch.cat((x1,x2),dim=2) # second dimension because of channels now in first
B, C, NP, LP = x.shape
x = x.reshape(B*C, NP, LP)

x1=model.contextBlockFunc(x1,64)
x2=model.contextBlockFunc(x2,64)
x=torch.cat((x1,x2),dim=1)

attn_basic = model.get_attention_maps(x) # Returns list of length of layers 
# Attn matrix is shape torch.Size([3, 16, 16])  batch size x target seq len x source seq length

# We are visualizing one per sample 