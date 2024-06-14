from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers import Mlp, DropPath


class LayerNorm(nn.LayerNorm):

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        return super(LayerNorm, self).forward(input.float())


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.vis = False

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        weights = attn if self.vis else None
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        return x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, weights = self.attn(x)
        x = h + self.drop_path1(self.ls1(x))

        h = x
        x = self.norm2(x)
        x = self.mlp(x)

        x = h + self.drop_path2(self.ls2(x))
        return x, weights


class SIMVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        init_values = kwargs.pop('init_values')

        self.extract_layers = kwargs.pop('extract_layers')

        super(SIMVisionTransformer, self).__init__(**kwargs)

        self.patch_size = kwargs['patch_size']

        drop_path_rate = kwargs['drop_path_rate']
        depth = kwargs['depth']
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'],
                qkv_bias=kwargs['qkv_bias'],
                init_values=init_values, norm_layer=kwargs['norm_layer'], drop_path=dpr[i])
            for i in range(kwargs['depth'])])

        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)

        del self.norm  # remove the original norm

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

    def forward(self, x):
        extract_layers = []

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        hidden_states = x

        for depth, blk in enumerate(self.blocks):
            hidden_states, _ = blk(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)
            if depth == len(self.blocks) - 1:
                outcome = hidden_states

        outcome = self.fc_norm(outcome)
        output = self.head(outcome[:, 0])

        return output, outcome[:, 0], extract_layers


def unetr_vit_base_patch16(**kwargs):
    model = SIMVisionTransformer(
        img_size=256, extract_layers=[3, 6, 9, 12], patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model