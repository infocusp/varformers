"""Module providing performer helper and related functions and classes."""

import math
import torch
from torch import nn
from einops import rearrange, repeat
from functools import partial

class PerformerAttention(nn.Module):
    def __init__(self, dim, heads, device,  
                 generalized_attention = False,
                 kernel_fn = nn.ReLU(), dropout = 0.1, 
                 qkv_bias = True, attn_out_bias = True, ortho_scaling = 0, 
                 auto_check_redraw = True, feature_redraw_interval = 1000
    ):
        super(PerformerAttention, self).__init__()
        
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_heads = dim // heads
        inner_dim = dim_heads * heads        
        nb_features = dim_heads // 2
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.generalized_attention = generalized_attention

        self.create_projection = partial(gaussian_orthogonal_random_matrix, 
                                         nb_rows = self.nb_features, nb_columns = self.dim_heads,
                                         scaling = self.ortho_scaling, device=device)
        
        
        self.register_buffer('projection_matrix', self.create_projection())

        if generalized_attention:
            self.generalized_attention = generalized_attention
            self.kernel_fn = kernel_fn

        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)
        
        self.feature_redraw_interval = feature_redraw_interval
        self.current_feature_interval = 0
        self.auto_check_redraw = auto_check_redraw
        
    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, x, mask):
                        
        b, n, _ = *x.shape,     
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        
        if mask is not None:
            v.masked_fill_(mask, 0.)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        
        device = q.device

        if self.generalized_attention:
            create_kernel = partial(simple_kernel, kernel_fn = self.kernel_fn,
                                    projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(exponential_kernel, projection_matrix = self.projection_matrix,
                                    device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)
        
        out = performer_attention(q, k, v)    
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        
        self.current_feature_interval += 1
        
        if self.auto_check_redraw and self.current_feature_interval >= self.feature_redraw_interval:
            self.projection_matrix = self.create_projection(device = device)
            self.current_feature_interval = 0
        
        return self.dropout(out)


def exponential_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash -  diag_data-
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

def simple_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    # if TORCH_GE_1_8_0:
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    # else:
    #     q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


# linear attention
def performer_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


