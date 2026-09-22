
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalEncoder(nn.Module):
    """Per-channel temporal CNN. Four conv blocks; first three downsample."""

    def __init__(self, out_channels=32, dropout=0.0):
        super().__init__()
        blocks = []
        in_ch = 1
        for i in range(4):
            layers = [
                nn.Conv1d(in_ch, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ]
            # 640 -> 80 in the paper, so only three pooling operations.
            if i < 3:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*layers))
            in_ch = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # x: [B, C, L]
        b, c, l = x.shape
        x = x.reshape(b * c, 1, l)
        for block in self.blocks:
            x = block(x)
        _, d, lt = x.shape
        return x.reshape(b, c, d, lt)  # [B, C, 32, L/8]


class _SpatialTransformer(nn.Module):
    """Transformer across measurement channels (radar voxels in the original)."""

    def __init__(self, num_channels, d_model=32, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.channel_pos = nn.Parameter(torch.zeros(1, max(1, num_channels), d_model))
        nn.init.normal_(self.channel_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def _pos(self, c):
        if c <= self.channel_pos.size(1):
            return self.channel_pos[:, :c]
        # Robust fallback if runtime channel count differs from configs.enc_in.
        repeat = (c + self.channel_pos.size(1) - 1) // self.channel_pos.size(1)
        return self.channel_pos.repeat(1, repeat, 1)[:, :c]

    def forward(self, temporal_features):
        # [B, C, 32, Lt] -> [B, C, 32]
        x = temporal_features.mean(dim=-1)
        x = x + self._pos(x.size(1))
        x = self.encoder(x)
        return self.norm(x)


class _TemporalExpansion(nn.Module):
    """Expand temporal features back toward the original sequence length."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, target_len):
        # x: [B, C, 32, Lt]
        b, c, d, lt = x.shape
        x = self.net(x.reshape(b * c, d, lt))
        if x.size(-1) != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.reshape(b, c, 4, target_len)


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size,
            dilation=dilation, padding=self.pad
        )

    def forward(self, x):
        y = self.conv(x)
        return y[..., :-self.pad] if self.pad > 0 else y


class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size=2, dilation=dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size=2, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        r = self.skip(x)
        y = self.drop(self.act(self.norm1(self.conv1(x))))
        y = self.drop(self.act(self.norm2(self.conv2(y))))
        return self.act(y + r)


class Model(nn.Module):
    """
    Transformer-style wrapper used by the LaserNet training framework.

    Inputs
    ------
    x_enc : [B, L, C]
    x_mark_enc, x_dec, x_mark_dec : accepted for API compatibility; unused.

    Output
    ------
    [B, pred_len, c_out]
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len

        channel_independence = bool(getattr(configs, "channel_independence", False))
        self.enc_in = 1 if channel_independence else int(configs.enc_in)
        self.c_out = 1 if channel_independence else int(configs.c_out)
        dropout = float(getattr(configs, "dropout", 0.1))

        self.temporal_encoder = _TemporalEncoder(out_channels=32, dropout=0.0)
        self.spatial_encoder = _SpatialTransformer(
            num_channels=self.enc_in,
            d_model=32,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            dropout=dropout,
        )
        self.temporal_expand = _TemporalExpansion()
        self.spatial_expand = nn.Linear(32, 4)

        # Nine exponentially dilated causal blocks: 1, 2, ..., 256.
        tcn = []
        in_ch = 4
        for i in range(9):
            tcn.append(_TCNBlock(in_ch, 8, dilation=2 ** i, dropout=dropout))
            in_ch = 8
        self.tcn = nn.Sequential(*tcn)
        self.head = nn.Conv1d(8, self.c_out, kernel_size=1)

    def forecast(self, x_enc):
        # [B, L, C] -> [B, C, L]
        x = x_enc.transpose(1, 2)
        target_len = x.size(-1)

        temporal = self.temporal_encoder(x)                 # [B,C,32,L/8]
        spatial = self.spatial_encoder(temporal)            # [B,C,32]
        temporal = self.temporal_expand(temporal, target_len)  # [B,C,4,L]
        spatial = self.spatial_expand(spatial).unsqueeze(-1)   # [B,C,4,1]

        # Original MMECG uses element-wise multiplication for feature fusion.
        fused = (temporal * spatial).mean(dim=1)            # [B,4,L]
        y = self.head(self.tcn(fused)).transpose(1, 2)      # [B,L,c_out]
        return y

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        y = self.forecast(x_enc)
        return y[:, -self.pred_len:, :]
