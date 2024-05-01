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
    def __init__(self, input_dim, embed_dim, num_heads):
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
        self.UqUk_proj = nn.Linear(embed_dim, 2*embed_dim, bias=False) 
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.Wqkv_proj.weight)
        nn.init.xavier_uniform_(self.Wo_proj.weight)
        nn.init.xavier_uniform_(self.UqUk_proj.weight)
        
    def forward(self, x, PE, PE_r):
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

        return Wo


class EncoderBlock(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, dim_ff, dropout=0.0):
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
                                                num_heads=num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(nn.Linear(embed_dim, dim_ff),
                                        nn.ReLU(),
                                        nn.Linear(dim_ff, embed_dim)
                                        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, PE, PE_r, mask=None):
        # Attention part
        attn_out = self.TUPE_attn(x, PE, PE_r)
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

    def forward(self, x, PE, PE_r, mask=None):
        for layer in self.layers:
            x = layer(x, PE, PE_r, mask=mask)
        return x

    def get_attention_maps(self, x, PE, PE_r, mask=None):
        attention_maps = []
        for idx, layer in enumerate(self.layers):
            print("Going into layer ", idx, "in TransformerEncoder")
            _, attn_map = layer.self_attn(x, PE, PE_r, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

################################# Transformer #################################
import torch.nn as nn

# https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16
class RelativePositionBias(nn.Module):
    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=8):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)
        
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (torch.tensor(n) < 0).to(torch.long) * num_buckets  
            n = torch.abs(torch.tensor(n))
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, qlen, klen):
        return self.compute_bias(qlen, klen)  # shape (1, num_heads, qlen, klen)

        
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

#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class RelativeTUPETransformer(pl.LightningModule):
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
        mask = None
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
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
       
        # Relative Learnable Positional Embedding 
        self.R_PE = RelativePositionBias(bidirectional=True, num_buckets=32,
                                         max_distance=128, n_heads=self.hparams.num_heads)
        
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            patch_size=self.hparams.patch_size,
            embed_dim=self.hparams.model_dim,
            dim_ff=2 * self.hparams.model_dim, 
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )
        
        # Output layer
        self.patches = int((((self.hparams.context_size/2)-self.hparams.patch_size)/self.hparams.step+1)*2)
        flattenOutSize= int(self.patches*self.hparams.model_dim)
        
        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(flattenOutSize, self.hparams.output_dim),
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
        PE = self.positional_encoding(x)
        PE_r = self.R_PE(self.patches, self.patches)
        
        #forward pass
        print("Going into input_net()")
        x = self.input_net(x)
        print("Going into transformer()")
        x = self.transformer(x, PE, PE_r, mask=self.hparams.mask) # Might need to do something different with mask 
        print("Going into output_net()")
        x=self.output_net(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x,y=batch
        x1,x2=x
        
        # Ensure splitting into blocks can be done exactly
        assert x1.shape[1]%self.hparams.patch_size==0 , print("x1 has wrong shape", x1.shape,self.hparams.patch_size)
        assert x2.shape[1]%self.hparams.patch_size==0, print("x2 has wrong shape", x2.shape,self.hparams.patch_size)
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