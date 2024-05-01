# Standard libraries
import math
import numpy as np

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

############################### Other parts ###################################

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


    def forward(self, x):
        return self.net(x)

################################### Encoder ###################################

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_ff, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_ff: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer - uses same dimension for k, q and v. 
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                               num_heads=num_heads,
                                               batch_first=True)

        # Two-layer MLP
        self.linear_net = nn.Sequential(nn.Linear(input_dim, dim_ff),
                                        nn.ReLU(),
                                        nn.Linear(dim_ff, input_dim)
                                        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, x, x)[0] # output is tuple therefore [0]
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

################################# Transformer #################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        (sine and cosine approach)
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class Transformer(pl.LightningModule):
    def __init__(
        self,
        context_size, 
        context_block,
        output_dim,
        model_dim,
        num_heads,
        num_layers,
        lr,
        warmup=100,
        max_iters=1000,
        dropout=0.0,
        input_dropout=0.0,
        mask = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        assert context_size % context_block == 0, "context_size must be divisible by context_block"
        
        self.metric = torch.nn.L1Loss()
        
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.context_block, self.hparams.model_dim)
        )
        
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_ff=2 * self.hparams.model_dim, 
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )
        
        # Output layer
        transFormerOutSize=int(self.hparams.context_size/self.hparams.context_block*self.hparams.model_dim)
        
        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(transFormerOutSize,self.hparams.output_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.output_dim,self.hparams.output_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.output_dim,self.hparams.output_dim)
            )
        
    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        self.lr_scheduler = CosineWarmupScheduler(optimizer, 
                                                  warmup=self.hparams.warmup, 
                                                  max_iters=self.hparams.max_iters)
        
        return {'optimizer': optimizer, "lr_scheduler": self.lr_scheduler}
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
        
    def contextBlockFunc(self,x,context_block):
        #returns a list of context blocks i.e. cuts the input into smaller blocks 
        #assumes input is (batch,sequence), returns (batch,sequence/context_block)
        #reshaping to (batch,context_block,sequence):
        x=x.reshape(x.shape[0],context_block,-1)
        #transposing to (batch,sequence,context_block)
        return torch.transpose(x,1,2)
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
    
    def forward(self, inputs):
        x1, x2 = inputs

        #input x1 and x2: returns as (batch,sequence,context_block)
        x1=self.contextBlockFunc(x1,self.hparams.context_block)
        x2=self.contextBlockFunc(x2,self.hparams.context_block)
        x=torch.cat((x1,x2),dim=1)

        #forward pass
        x = self.input_net(x)
        x = self.positional_encoding(x)
        
        x = self.transformer(x, mask=self.hparams.mask) # Might need to do something different with mask 
        x=self.output_net(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x,y=batch
        x1,x2=x
        
        # Ensure splitting into blocks can be done exactly
        assert x1.shape[1]%self.hparams.context_block==0 , print("x1 has wrong shape", x1.shape,self.hparams.context_block)
        assert x2.shape[1]%self.hparams.context_block==0, print("x2 has wrong shape", x2.shape,self.hparams.context_block)
        assert y.shape[1]==self.hparams.output_dim, print("y has wrong shape", y.shape,self.hparams.output_dim)
        
        pred = self.forward(x)
        loss = F.mse_loss(pred, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        
        pred = self.forward(x)
        loss = self.metric(pred, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    
        return loss
    
    