import torch
from torch import nn, einsum
from math import pi, log
from einops import rearrange, repeat
#Code adapted from https://github.com/lucidrains/perceiver-pytorch

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_bands, base = base, device = device, dtype = dtype)
    scales = rearrange(scales, 's -> () () () s')

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


# My adaptations

def positional_encode_cloud(xyz,max_freq=256, num_bands = 6, base = 2):
    b, *axis, _, device = *xyz.shape, xyz.device
    axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
    pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
    enc_pos = fourier_encode(pos, max_freq, num_bands, base = 2)
    enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
    enc_pos = repeat(enc_pos, '... -> b ...', b = b)


    return enc_pos
class AttentionControlledOut(nn.Module):
    def __init__(self,out_dim,query_dim, context_dim, heads, dim_head, dropout):
        super().__init__()
        self.attention = Attention(query_dim, context_dim, heads , dim_head , dropout )
        self.lin = nn. Linear(query_dim,out_dim)
    
    def forward(self, x, context = None, mask = None):
        return self.lin(self.attention(x, context = context, mask = mask))

    


get_cross_attn = lambda out_dim,query_dim, context_dim, heads, dim_head, dropout: PreNorm(query_dim, AttentionControlledOut(out_dim,query_dim, context_dim, heads, dim_head, dropout))
if __name__ == '__main__':
    test_data = torch.rand((10,2048,3))*2 -1
    test_data = test_data.reshape((10,64,32,3))
    enc_pos = positional_encode_cloud(test_data)
    cross_attn = get_cross_attn(49,50,256,1,64,0.0).cuda()
    prev_attn_and_curr_points = torch.randn((10,2048,50)).cuda()
    context_data = torch.randn((10,256,256)).cuda()
    emb = cross_attn(prev_attn_and_curr_points,context = context_data)
    