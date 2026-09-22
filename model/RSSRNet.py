
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * gate


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.ca = _ChannelAttention(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.ca(self.net(x)))


class _SeparatorBlock(nn.Module):
    """Self-attention separator operating on flattened time-frequency tokens."""

    def __init__(self, d_model=128, nhead=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, C, F, T]
        b, c, f, t = x.shape
        z = x.flatten(2).transpose(1, 2)  # [B, F*T, C]
        q = self.norm1(z)
        z = z + self.attn(q, q, q, need_weights=False)[0]
        z = z + self.ffn(self.norm2(z))
        return z.transpose(1, 2).reshape(b, c, f, t)


class _CrossChannelFusion(nn.Module):
    """CCA-style fusion of decoder and encoder skip features."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ca = _ChannelAttention(in_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, decoder_x, skip_x):
        if decoder_x.shape[-2:] != skip_x.shape[-2:]:
            decoder_x = F.interpolate(
                decoder_x, size=skip_x.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([decoder_x, skip_x], dim=1)
        return self.fuse(self.ca(x))


def _safe_n_fft(seq_len, requested=64):
    n = min(int(requested), int(seq_len))
    # torch.stft is most convenient with a power-of-two FFT size.
    return max(8, 2 ** int(math.floor(math.log2(max(8, n)))))


class Model(nn.Module):
    """
    LaserNet-compatible RSSRNet reproduction.

    Input:  x_enc [B, L, C]
    Output: [B, pred_len, c_out]

    Notes:
    - This is the ECG branch only; the original RSSRNet jointly reconstructs
      respiration and ECG.
    - It predicts real/imaginary STFT components and applies iSTFT internally.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        channel_independence = bool(getattr(configs, "channel_independence", False))
        self.enc_in = 1 if channel_independence else int(configs.enc_in)
        self.c_out = 1 if channel_independence else int(configs.c_out)
        dropout = float(getattr(configs, "dropout", 0.1))

        seq_len = int(getattr(configs, "seq_len", 512))
        self.n_fft = _safe_n_fft(seq_len, getattr(configs, "rssr_n_fft", 64))
        self.hop_length = int(getattr(configs, "rssr_hop_length", self.n_fft // 4))
        self.win_length = self.n_fft
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

        # Complex STFT is represented as two channels (real + imaginary) per input.
        in_spec_ch = 2 * self.enc_in
        self.stem = _ConvBlock(in_spec_ch, 32, stride=1, dropout=dropout)
        self.enc1 = _ConvBlock(32, 64, stride=2, dropout=dropout)
        self.enc2 = _ConvBlock(64, 128, stride=2, dropout=dropout)

        self.separator = nn.Sequential(
            _SeparatorBlock(128, nhead=4, d_ff=256, dropout=dropout),
            _SeparatorBlock(128, nhead=4, d_ff=256, dropout=dropout),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.cca1 = _CrossChannelFusion(64 + 64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.cca2 = _CrossChannelFusion(32 + 32, 32)

        # Two channels per ECG output: real and imaginary STFT components.
        self.spec_head = nn.Conv2d(32, 2 * self.c_out, kernel_size=1)

    def _stft(self, x):
        # x: [B, C, L]
        b, c, l = x.shape
        z = torch.stft(
            x.reshape(b * c, l),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x.dtype),
            center=True,
            return_complex=True,
        )
        f, t = z.shape[-2:]
        z = z.reshape(b, c, f, t)
        return torch.cat([z.real, z.imag], dim=1)  # [B,2C,F,T]

    def _istft(self, spec, length):
        # spec: [B, 2*c_out, F, T]
        b, _, f, t = spec.shape
        spec = spec.reshape(b, self.c_out, 2, f, t)
        z = torch.complex(spec[:, :, 0], spec[:, :, 1])
        y = torch.istft(
            z.reshape(b * self.c_out, f, t),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=spec.dtype),
            center=True,
            length=length,
        )
        return y.reshape(b, self.c_out, length)

    def forecast(self, x_enc):
        # [B,L,C] -> [B,C,L]
        x = x_enc.transpose(1, 2)
        length = x.size(-1)
        spec = self._stft(x)

        s0 = self.stem(spec)       # [B,32,F,T]
        s1 = self.enc1(s0)         # [B,64,F/2,T/2]
        z = self.enc2(s1)          # [B,128,F/4,T/4]
        z = self.separator(z)

        z = F.gelu(self.up1(z))
        z = self.cca1(z, s1)
        z = F.gelu(self.up2(z))
        z = self.cca2(z, s0)
        z = self.spec_head(z)

        # Match the original STFT grid exactly before inverse transform.
        if z.shape[-2:] != spec.shape[-2:]:
            z = F.interpolate(z, size=spec.shape[-2:], mode="bilinear", align_corners=False)

        y = self._istft(z, length=length).transpose(1, 2)  # [B,L,c_out]
        return y

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        y = self.forecast(x_enc)
        return y[:, -self.pred_len:, :]
