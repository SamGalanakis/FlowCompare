import torch
from torch import nn, einsum
from math import pi, log
from einops import rearrange, repeat
from functools import wraps
import torch.nn.functional as F
# Code adapted from https://github.com/lucidrains/perceiver-pytorch


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(
            context_dim) if exists(context_dim) else None

    def forward(self, x, context):
        x = self.norm(x)

        if exists(self.norm_context):
            context = context
            normed_context = self.norm_context(context)
        else:
            normed_context = context

        return self.fn(x, normed_context)







class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)





class AttentionControlledOut(nn.Module):
    def __init__(self, out_dim, query_dim, context_dim, heads, dim_head, dropout):
        super().__init__()
        self.attention = AttentionMine(query_dim, context_dim, heads, dim_head)
        self.lin = nn. Linear(self.attention.inner_dim, out_dim)

    def forward(self, x, context=None, mask=None):
        return self.lin(self.attention(x, context=context, mask=mask))


class AttentionMine(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head,save_attn_weights = False):
        super().__init__()
        self.save_attn_weights = save_attn_weights
        self.inner_dim = dim_head * heads
        self.scale = self.inner_dim ** -0.5
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        attn_weights = F.softmax(torch.matmul(
            q, k.transpose(1, 2))*self.scale, dim=-1)
        attn = torch.matmul(attn_weights, v)
        if self.save_attn_weights:
            self.last_attn_weights = attn_weights.cpu()
        return attn


def get_cross_attn(out_dim, query_dim, context_dim, heads, dim_head, dropout): return PreNorm(
    query_dim, AttentionControlledOut(out_dim, query_dim, context_dim, heads, dim_head, dropout))



