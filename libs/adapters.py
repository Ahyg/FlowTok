import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterIn(nn.Module):
    """
    Map arbitrary input channels/resolution to tokenizer-expected channels/resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        mid_channels: int = 32,
        num_blocks: int = 3,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(max(int(num_blocks), 1)):
            c_out = mid_channels
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(1, c_out),
                    nn.SiLU(),
                )
            )
            c_in = c_out
        self.blocks = nn.ModuleList(layers)
        # Final linear projection without norm/activation.
        self.out_proj = nn.Conv2d(c_in, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor, target_size: int) -> torch.Tensor:
        # x: [B*T, C_raw, H_raw, W_raw]
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)
        if x.shape[-2] != target_size or x.shape[-1] != target_size:
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        #return torch.clamp(x, 0.0, 1.0)
        return x


class AdapterOut(nn.Module):
    """
    Map tokenizer decoded images to dataset target channels/resolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        mid_channels: int = 16,
        num_blocks: int = 2,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(max(int(num_blocks), 1)):
            c_out = mid_channels
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(1, c_out),
                    nn.SiLU(),
                )
            )
            c_in = c_out
        self.blocks = nn.ModuleList(layers)
        # Final linear projection without norm/activation.
        self.out_proj = nn.Conv2d(c_in, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        # x: [B*T, C_in, H_dec, W_dec]
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(
                x,
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )
        #return torch.clamp(x, 0.0, 1.0)
        return x

