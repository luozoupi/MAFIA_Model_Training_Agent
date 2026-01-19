# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernels for window partition / reverse (Swin-V2)
# ------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void window_partition_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ output,
        const int B, const int H, const int W, const int C,
        const int window_size,
        const int num_win_h, const int num_win_w) {

    const long long total = (long long)B * num_win_h * num_win_w *
                            window_size * window_size * C;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    long long tmp = idx;

    const int c     = tmp % C;               tmp /= C;
    const int wj    = tmp % window_size;     tmp /= window_size;
    const int wi    = tmp % window_size;     tmp /= window_size;
    const int win_w = tmp % num_win_w;       tmp /= num_win_w;
    const int win_h = tmp % num_win_h;       tmp /= num_win_h;
    const int b     = tmp;

    const int in_h = win_h * window_size + wi;
    const int in_w = win_w * window_size + wj;
    const long long in_idx =
        (((long long)b * H + in_h) * W + in_w) * C + c;

    output[idx] = input[in_idx];
}

template <typename scalar_t>
__global__ void window_reverse_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ output,
        const int B, const int H, const int W, const int C,
        const int window_size,
        const int num_win_h, const int num_win_w) {

    const long long total = (long long)B * num_win_h * num_win_w *
                            window_size * window_size * C;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    long long tmp = idx;

    const int c     = tmp % C;               tmp /= C;
    const int wj    = tmp % window_size;     tmp /= window_size;
    const int wi    = tmp % window_size;     tmp /= window_size;
    const int win_w = tmp % num_win_w;       tmp /= num_win_w;
    const int win_h = tmp % num_win_h;       tmp /= num_win_h;
    const int b     = tmp;

    const int out_h = win_h * window_size + wi;
    const int out_w = win_w * window_size + wj;
    const long long out_idx =
        (((long long)b * H + out_h) * W + out_w) * C + c;

    output[out_idx] = input[idx];
}

// -----------------------------------------------------------
// Launcher functions
// -----------------------------------------------------------
torch::Tensor window_partition_cuda(torch::Tensor input, int window_size) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dim() == 4, "input must be (B, H, W, C)");

    const int B = input.size(0);
    const int H = input.size(1);
    const int W = input.size(2);
    const int C = input.size(3);
    TORCH_CHECK(H % window_size == 0 && W % window_size == 0,
                "H and W must be divisible by window_size");

    const int num_win_h = H / window_size;
    const int num_win_w = W / window_size;
    const int total_win = B * num_win_h * num_win_w;

    auto output = torch::empty({total_win, window_size, window_size, C},
                               input.options());

    const long long total_elems =
        (long long)total_win * window_size * window_size * C;
    const int threads = 256;
    const int blocks  = (total_elems + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "window_partition_cuda_launch", ([&] {
            window_partition_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                B, H, W, C,
                window_size, num_win_h, num_win_w);
        }));

    return output;
}

torch::Tensor window_reverse_cuda(torch::Tensor input,
                                  int window_size,
                                  int H, int W) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dim() == 4, "input must be (nWin*B, ws, ws, C)");

    const int C = input.size(3);
    const int num_win_h = H / window_size;
    const int num_win_w = W / window_size;
    const int total_win = input.size(0);
    const int B = total_win / (num_win_h * num_win_w);

    auto output = torch::empty({B, H, W, C}, input.options());

    const long long total_elems =
        (long long)total_win * window_size * window_size * C;
    const int threads = 256;
    const int blocks  = (total_elems + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "window_reverse_cuda_launch", ([&] {
            window_reverse_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                B, H, W, C,
                window_size, num_win_h, num_win_w);
        }));

    return output;
}
"""

# ------------------------------------------------------------------
# C++ prototypes so that main.cpp sees the functions
# ------------------------------------------------------------------
cpp_src = r"""
#include <torch/extension.h>
torch::Tensor window_partition_cuda(torch::Tensor input, int window_size);
torch::Tensor window_reverse_cuda(torch::Tensor input,
                                  int window_size,
                                  int H, int W);
"""

# ------------------------------------------------------------------
# compile & load
# ------------------------------------------------------------------
window_ops = load_inline(
    name="window_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["window_partition_cuda", "window_reverse_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# High-level autograd-aware python wrappers
# ------------------------------------------------------------------
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def _cpu_window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = (x.permute(0, 1, 3, 2, 4, 5)
                 .contiguous()
                 .view(-1, window_size, window_size, C))
    return windows

def _cpu_window_reverse(windows: torch.Tensor,
                        window_size: int,
                        H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = (windows.view(B, H // window_size, W // window_size,
                      window_size, window_size, -1)
               .permute(0, 1, 3, 2, 4, 5)
               .contiguous()
               .view(B, H, W, -1))
    return x

class _WindowPartitionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, window_size: int):
        ctx.window_size = window_size
        ctx.input_shape = x.shape
        if x.is_cuda:
            y = window_ops.window_partition_cuda(
                x.contiguous(), window_size)
        else:
            y = _cpu_window_partition(x, window_size)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        window_size = ctx.window_size
        B, H, W, C = ctx.input_shape
        if grad_output.is_cuda:
            grad_input = window_ops.window_reverse_cuda(
                grad_output.contiguous(), window_size, H, W)
        else:
            grad_input = _cpu_window_reverse(
                grad_output, window_size, H, W)
        return grad_input, None

class _WindowReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, windows: torch.Tensor,
                window_size: int, H: int, W: int):
        ctx.window_size = window_size
        ctx.H, ctx.W = H, W
        if windows.is_cuda:
            y = window_ops.window_reverse_cuda(
                windows.contiguous(), window_size, H, W)
        else:
            y = _cpu_window_reverse(windows, window_size, H, W)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        window_size = ctx.window_size
        if grad_output.is_cuda:
            grad_input = window_ops.window_partition_cuda(
                grad_output.contiguous(), window_size)
        else:
            grad_input = _cpu_window_partition(
                grad_output, window_size)
        return grad_input, None, None, None

def window_partition(x: torch.Tensor, window_size: int):
    return _WindowPartitionFn.apply(x, window_size)

def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int):
    return _WindowReverseFn.apply(windows, window_size, H, W)

# ------------------------------------------------------------------
# Swin-Transformer V2 implementation (only window ops modified)
# ------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))),
            requires_grad=True)

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False))

        relative_coords_h = torch.arange(
            - (self.window_size[0] - 1),
            self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(
            - (self.window_size[1] - 1),
            self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])
        ).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[..., 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[..., 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[..., 0] /= (self.window_size[0] - 1)
            relative_coords_table[..., 1] /= (self.window_size[1] - 1)

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * \
            torch.log2(torch.abs(relative_coords_table) + 1.) / np.log2(8)
        self.register_buffer("relative_coords_table",
                             relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()
        relative_coords[..., 0] += self.window_size[0] - 1
        relative_coords[..., 1] += self.window_size[1] - 1
        relative_coords[..., 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias, requires_grad=False),
                 self.v_bias))
        qkv = torch.nn.functional.linear(x, self.qkv.weight, qkv_bias)
        qkv = (qkv.reshape(B_, N, 3, self.num_heads, -1)
                     .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (torch.nn.functional.normalize(q, dim=-1) @
                torch.nn.functional.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(
            self.logit_scale.to(x.device),
            max=torch.log(torch.tensor(1. / 0.01,
                                       device=x.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(
            relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N) +
                    mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(
                img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = (mask_windows.unsqueeze(1) -
                         mask_windows.unsqueeze(2))
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(
                    attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(
            shifted_x, self.window_size)
        x_windows = x_windows.view(-1,
                                   self.window_size * self.window_size,
                                   C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        x = self.norm(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, input_resolution, num_heads,
                window_size, 0 if i % 2 == 0 else window_size // 2,
                mlp_ratio, qkv_bias, drop,
                attn_drop, drop_path[i] if
                isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size)
            for i in range(depth)
        ])

        self.downsample = (downsample(input_resolution, dim,
                                      norm_layer=norm_layer)
                           if downsample is not None else None)

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0],
                                   img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 pretrained_window_sizes=[0, 0, 0, 0]):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans,
            embed_dim, norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):
                              sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=(PatchMerging if
                            (i_layer < self.num_layers - 1) else None),
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (nn.Linear(self.num_features, num_classes)
                     if num_classes > 0 else nn.Identity())

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

batch_size = 5
image_size = 224
def get_inputs():
    return [torch.rand(batch_size, 3, image_size, image_size).cuda()]
def get_init_inputs():
    return []
# </complete ModelNew code>
