import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if return_attn:
            # explicit path so we can keep the attention maps
            attn = ((q @ k.transpose(-2, -1)) * self.head_dim ** -0.5).softmax(dim=-1)
            out = attn @ v
        else:
            attn = None
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out), attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, return_attn=False):
        a, attn = self.attn(self.norm1(x), return_attn)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, attn


class ViT(nn.Module):
    def __init__(self, img_size=32, patch=4, dim=192, depth=6, heads=3, mlp_ratio=4.0,
                 num_classes=10, in_ch=3):
        super().__init__()
        assert img_size % patch == 0
        self.patch = patch
        self.num_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init()

    def _init(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def tokens(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos_embed

    def forward(self, x, return_attn=False):
        x = self.tokens(x)
        attns = []
        for blk in self.blocks:
            x, a = blk(x, return_attn)
            if return_attn:
                attns.append(a)
        logits = self.head(self.norm(x)[:, 0])
        return (logits, attns) if return_attn else logits


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    # ResNet-18 layout, CIFAR stem (3x3 conv, stride 1, no max-pool)
    def __init__(self, layers=(2, 2, 2, 2), width=0.5, num_classes=10):
        super().__init__()
        chans = [int(c * width) for c in (64, 128, 256, 512)]
        self.stem = nn.Sequential(nn.Conv2d(3, chans[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(chans[0]), nn.ReLU(inplace=True))
        stages, in_ch = [], chans[0]
        for i, (n, c) in enumerate(zip(layers, chans)):
            blocks = []
            for j in range(n):
                blocks.append(BasicBlock(in_ch, c, (1 if i == 0 else 2) if j == 0 else 1))
                in_ch = c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(in_ch, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


MODEL_NAMES = ("vit_p4", "vit_p8", "resnet18_w05")


def build_model(name):
    if name == "vit_p4":
        return ViT(patch=4)
    if name == "vit_p8":
        return ViT(patch=8)
    if name == "resnet18_w05":
        return ResNetCIFAR(width=0.5)
    raise ValueError(f"unknown model {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_groups(model, weight_decay):
    # no decay on biases, norm params, cls token and pos embedding
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "pos_embed" in n or "cls_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]
