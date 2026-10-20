"""The three architectures compared in Lab 3, exactly as specified in the manual.

One addition: the attention model gets a learnable positional embedding (16 x 64). Without
it, self-attention followed by mean pooling cannot tell which patch came from where; the
manual is silent on this, and an ablation without it is reported alongside.
"""
import torch
from torch import nn

PATCH = 7
N_PATCHES = (28 // PATCH) ** 2   # 16
D_MODEL = 64


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchTransformer(nn.Module):
    """16 patches of 7x7 -> Linear 49->64 (+ positional embedding) -> 2 encoder layers
    (d_model 64, 4 heads, FFN 128) -> mean pool over patches -> Linear 64->10."""

    def __init__(self, positional: bool = True):
        super().__init__()
        self.proj = nn.Linear(PATCH * PATCH, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, N_PATCHES, D_MODEL)) if positional else None
        if self.pos is not None:
            nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=4, dim_feedforward=128,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(D_MODEL, 10)

    @staticmethod
    def patchify(x):
        """(B, 1, 28, 28) -> (B, 16, 49), patches in row-major order."""
        b = x.shape[0]
        return x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH).reshape(b, N_PATCHES, PATCH * PATCH)

    def forward(self, x):
        tokens = self.proj(self.patchify(x))
        if self.pos is not None:
            tokens = tokens + self.pos
        return self.head(self.encoder(tokens).mean(dim=1))


MODEL_FACTORIES = {"MLP": MLP, "CNN": CNN, "Attention": PatchTransformer}


def build_models():
    return {name: factory() for name, factory in MODEL_FACTORIES.items()}
