
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Residual1DBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, dropout=0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return F.relu(x + y, inplace=True)


class _ImprovedResStack(nn.Module):
    def __init__(self, channels, num_blocks=6, dropout=0.1):
        super().__init__()
        # Increasing dilation enlarges the temporal receptive field while
        # preserving sequence length.
        dilations = [1, 1, 2, 2, 4, 4]
        if num_blocks != 6:
            dilations = [2 ** min(i // 2, 4) for i in range(num_blocks)]
        self.blocks = nn.Sequential(*[
            _Residual1DBlock(channels, kernel_size=7, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x):
        return self.blocks(x)


class Model(nn.Module):
    """
    LaserNet-compatible RadarNet reconstruction network.

    Input:  x_enc [B, L, C]
    Output: [B, pred_len, c_out]
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        channel_independence = bool(getattr(configs, "channel_independence", False))
        self.enc_in = 1 if channel_independence else int(configs.enc_in)
        self.c_out = 1 if channel_independence else int(configs.c_out)
        dropout = float(getattr(configs, "dropout", 0.1))
        width = int(getattr(configs, "radarnet_width", 64))
        num_blocks = int(getattr(configs, "radarnet_blocks", 6))

        # Conv1d coarse reconstruction stage.
        self.encoder = nn.Sequential(
            nn.Conv1d(self.enc_in, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            _Residual1DBlock(width, kernel_size=7, dilation=1, dropout=dropout),
            _Residual1DBlock(width, kernel_size=7, dilation=2, dropout=dropout),
        )
        self.coarse_head = nn.Conv1d(width, self.c_out, kernel_size=7, padding=3)

        # Map the coarse ECG back into feature space, then use ResNet blocks to
        # learn a residual correction (signal amplification/refinement).
        self.coarse_embed = nn.Conv1d(self.c_out, width, kernel_size=3, padding=1)
        self.refiner = _ImprovedResStack(width, num_blocks=num_blocks, dropout=dropout)
        self.refine_head = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(width // 2, self.c_out, kernel_size=3, padding=1),
        )

    def forecast(self, x_enc):
        x = x_enc.transpose(1, 2)                 # [B,C,L]
        feat = self.encoder(x)                    # [B,W,L]
        coarse = self.coarse_head(feat)           # [B,c_out,L]
        refined_feat = self.refiner(feat + self.coarse_embed(coarse))
        correction = self.refine_head(refined_feat)
        y = coarse + correction                   # residual refinement
        return y.transpose(1, 2)                  # [B,L,c_out]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        y = self.forecast(x_enc)
        return y[:, -self.pred_len:, :]
