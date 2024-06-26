"""
Non-TUPE multihead attention is inspired by: 
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    
Relative positional encoding is a slightly modified version of: 
    https://huggingface.co/transformers/v3.2.0/_modules/transformers/modeling_t5.html
"""

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
def tupe_product(q, k):
    d = q.size()[-1]
    attn = torch.matmul(q, k.transpose(-2, -1))
    return attn / math.sqrt(2*d)

class TUPEMultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, TUPE=False):
        """
        Input: 
            input_dim: Dimensionality of input 
            embed_dim: Embedding dimension 
            num_heads: Number of heads 
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim=embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.Wqkv_proj = nn.Linear(input_dim, 3*embed_dim, bias=False) # 3 because we have 3 projection matrices
        self.Wo_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        if TUPE: 
            self.UqUk_proj = nn.Linear(embed_dim, 2*embed_dim, bias=False) 
        
        self._reset_parameters(TUPE)
    
    def _reset_parameters(self, TUPE):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.Wqkv_proj.weight)
        nn.init.xavier_uniform_(self.Wo_proj.weight)
        if TUPE: 
            nn.init.xavier_uniform_(self.UqUk_proj.weight)
        
    def forward(self, x, PE_r, PE=None):
        """
        Take in query, key, value i.e. x, x, x. 
        Return:  Tuple(tensor, optional(tensor))
            attention_output [batch_size, no_patches input_dim]
        
        for h heads it: 
            Parameters: 
                - Abs PE is shared across heads
                - Projection matrices U^Q and U^K are different across heads 
                but the same for the layers 
                (Thus, the last term encoding position needs only be calcuted 
                 for the first layer and then just added)
        """
        if not isinstance(PE, type(None)):
    		# TUPE 
            batch_size, patch_length, input_dim = x.size()
            head_dim = self.embed_dim // self.num_heads
            # First project x into q, k, and v i.e. multiply 
            qkv = self.Wqkv_proj(x) # torch.Size([10, 16, 64]) -> torch.Size([10, 16, 192])
            PE_term = self.UqUk_proj(PE)
            
            # Separate U_q and U_k from linear output 
            PE_term = PE_term.reshape(batch_size, -1, self.num_heads, 2*head_dim)
            # patches are divided across heads to be processed i.e. 16 patches of length 4 are processed on 16 different heads
            # Is that correctly understood? 
            PE_term = PE_term.permute(0, 2, 1, 3) # [Batch, Head, no patches, head_dim]
            Uq, Uk= PE_term.chunk(2, dim=-1)
            
            # Separate Q, K, V from linear output
            qkv = qkv.reshape(batch_size, -1, self.num_heads, 3*head_dim)
            # patches are divided across heads to be processed i.e. 16 patches of length 4 are processed on 16 differen heads
            # Is that correctly understood? 
            qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, no pathces, head_dim]
            q, k, v = qkv.chunk(3, dim=-1)
            
            PE_attn = tupe_product(Uq, Uk)
            word_attn = tupe_product(q, k)
    
            attention = nn.functional.softmax(PE_attn + word_attn + PE_r, dim=-1)
            values = torch.matmul(attention, v)
    
            values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
            values = values.reshape(batch_size, patch_length, self.num_heads*q.size()[-1])
            Wo = self.Wo_proj(values)
        else: 
            batch_size, patch_length, input_dim = x.size()
            head_dim = self.embed_dim // self.num_heads
            # First project x into q, k, and v i.e. multiply 
            qkv = self.Wqkv_proj(x) # torch.Size([10, 16, 64]) -> torch.Size([10, 16, 192])

            # Separate Q, K, V from linear output
            qkv = qkv.reshape(batch_size, -1, self.num_heads, 3*head_dim)
            # patches are divided across heads to be processed i.e. 16 patches of length 4 are processed on 16 differen heads
            qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, no pathces, head_dim]
            q, k, v = qkv.chunk(3, dim=-1)

            word_attn = tupe_product(q, k)
    
            attention = nn.functional.softmax(word_attn + PE_r, dim=-1)
            values = torch.matmul(attention, v)
    
            values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
            values = values.reshape(batch_size, patch_length, self.num_heads*q.size()[-1])
            Wo = self.Wo_proj(values)
        return Wo


class EncoderBlock(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, dim_ff, dropout=0.0, TUPE_=False):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_ff: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer - uses same dimension for k, q and v. 
        self.TUPE_attn = TUPEMultiheadAttention(input_dim=embed_dim,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                TUPE =TUPE_)

        # Two-layer MLP
        self.linear_net = nn.Sequential(nn.Linear(embed_dim, dim_ff),
                                        nn.ReLU(),
                                        nn.Linear(dim_ff, embed_dim)
                                        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, PE_r, PE):
        # Attention part
        attn_out = self.TUPE_attn(x, PE_r, PE)
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

    def forward(self, x, PE_r, PE=None):
        for layer in self.layers:
            x = layer(x, PE_r, PE)
        return x


################################# Transformer #################################
class ALiBi(nn.Module):
    def __init__(self, no_patches, n_heads=8):
        super().__init__()
        self.no_patches = no_patches 
        self.n_heads = n_heads
        
        cols = torch.arange(self.no_patches, dtype=torch.long)[:, None]
        rows = torch.arange(self.no_patches, dtype=torch.long)[None, :]
        relative_position = rows - cols 
        
        RP_heads = relative_position[None,:,:].expand(self.n_heads, -1, -1)
        ratio = 8/self.n_heads
        scalars = torch.tensor([1/2**i for i in np.arange(ratio, 8+ratio, ratio)], dtype=torch.float32)
        
        RP_heads = scalars[:, None, None] * RP_heads
        
        self.register_buffer("PE_r", RP_heads, persistent=False)
        
    def forward(self, batch_size):
        return self.PE_r[None,:, :,:].expand(batch_size,-1, -1, -1)

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
        return self.pe[:, : x.size(1)].repeat(x.size(0), 1, 1)


class ALiBiTransformer(pl.LightningModule):
    def __init__(
        self,
        context_size, 
        patch_size,
        step,
        output_dim,
        model_dim,
        num_heads,
        num_layers,
        lr,
        warmup=100,
        max_iters=1000,
        dropout=0.0,
        input_dropout=0.0,
        TUPE = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        assert context_size % patch_size == 0, "context_size must be divisible by context_block"
        
        self.metric = torch.nn.L1Loss()
        
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.patch_size, self.hparams.model_dim)
        )
        
        # Absolute PE 
        if TUPE:
            self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        
        patches = int((((self.hparams.context_size/2)-self.hparams.patch_size)/self.hparams.step+1)*2)
        self.ALiBi = ALiBi(no_patches=patches, n_heads = self.hparams.num_heads)
        
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            patch_size=self.hparams.patch_size,
            embed_dim=self.hparams.model_dim,
            dim_ff=2 * self.hparams.model_dim, 
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            TUPE_=self.hparams.TUPE
        )
        
        # Output layer
        flattenOutSize= int(patches*self.hparams.model_dim)
        
        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(flattenOutSize, self.hparams.output_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.output_dim,self.hparams.output_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.output_dim,self.hparams.output_dim)
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        self.lr_scheduler = CosineWarmupScheduler(optimizer, 
                                                  warmup=self.hparams.warmup, 
                                                  max_iters=self.hparams.max_iters)
        
        return {'optimizer': optimizer, "lr_scheduler": self.lr_scheduler}
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
        
    def contextBlockFunc(self, x):
        #returns a list of patches i.e. cuts the input into smaller patches 
        #assumes input is (batch,sequence)
        x = x.unfold(dimension = 1, size = self.hparams.patch_size, 
                     step = self.hparams.step) # batch_size x no. patches x stride
        return x
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, inputs):
        x1, x2 = inputs

        #input x1 and x2: returns as (batch,sequence,patch_size)
        x1=self.contextBlockFunc(x1)
        x2=self.contextBlockFunc(x2)
        x=torch.cat((x1,x2),dim=1)
        if self.hparams.TUPE:
            PE = self.positional_encoding(x)
        batch_size, _, _ = x.shape
        PE_ALiBi = self.ALiBi(batch_size)
        #forward pass
        x = self.input_net(x)
        if self.hparams.TUPE:
            x = self.transformer(x, PE_ALiBi, PE) 
        else: 
            x = self.transformer(x, PE_ALiBi, PE=None)
        x=self.output_net(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x,y=batch
        x1,x2=x
        
        # Ensure splitting into blocks can be done exactly
        assert x1.shape[1]%self.hparams.patch_size==0 , print("x1 has wrong shape", x1.shape,self.hparams.patch_size)
        assert x2.shape[1]%self.hparams.patch_size==0, print("x2 has wrong shape", x2.shape,self.hparams.patch_size)
        assert y.shape[1]==self.hparams.output_dim, print("y has wrong shape", y.shape,self.hparams.output_dim)
        
        print("Into forward")
        pred = self.forward(x)
        print("Getting loss")
        loss = F.mse_loss(pred, y)
        
        print("logging")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        
        pred = self.forward(x)
        loss = self.metric(pred, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    
        return loss