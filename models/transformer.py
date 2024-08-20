import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import copy
import math
from functools import partial
from performer import PerformerAttention
from longformer import LongformerSelfAttention, LongformerConfig

def build_model(model_type, vocab_size, d_model, h, d_ff,
                N, dropout, decoder_layers, device):
    
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    if model_type=="transformer":
        attn = MultiHeadedAttention(h, d_model)
    elif model_type=="performer":
        attn = PerformerAttention(d_model, h, device, dropout=dropout)
    elif model_type=="longformer":
        config = LongformerConfig(hidden_size = d_model,
                 num_attention_heads = h,
                 attention_probs_dropout_prob=dropout,
                 attention_window = [8]*N, 
                 attention_dilation = [1]*N)
        
        attn = partial(LongformerSelfAttention, config)
        
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)    
    
    encoder_list = []
    for i in range(N):
        if model_type=="longformer":
            encoder_layer = EncoderLayer(d_model, c(attn(i)), c(ff), dropout)
        else:
            encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        encoder_list.append(encoder_layer)
    
    encoder_list = nn.ModuleList(encoder_list)
    model = Transformer(
        Encoder(encoder_list),
        ClsDecoder(*decoder_layers),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position))
    ) 

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model
    
class Transformer(nn.Module):
    """
    A standard Encoder Decoder architecture for classification architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder        
        self.decoder = decoder
        

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask)[:, 0, :])

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, cls_emb):
        return self.decoder(cls_emb)
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layers):
        super(Encoder, self).__init__()        
        self.layers = layers
        self.norm = LayerNorm(layers[0].size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, mask))
        return self.sublayer[1](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        "Implements Figure 2"
        
        query, key, value = (x, x, x)
        
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """
    
    def __init__(self, *args):
        
        super().__init__()        
        # module list        
        self._decoder = nn.ModuleList()
                
        n = len(args)
        
        for i in range(n-2):            
            self._decoder.append(nn.Linear(args[i], args[i+1]))
            self._decoder.append(nn.ReLU())

        self.out_layer = nn.Linear(args[n-2], args[n-1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)        
        x = self.out_layer(x)
        return self.sigmoid(x)

