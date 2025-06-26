
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


all_android_domains = set([
    'google.com',
    'googleapis.com',
    'gstatic.com',
    'googleadservices.com',
    'youtube.com',
    'app-measurement.com',
    'android.com',
    'google-analytics.com',
    'ntp.org',
    'googleusercontent.com',
    'doubleclick.net',
    'crashlytics.com',
    'googletagmanager.com',
    'tenor.com',
    'googlesyndication.com',
    'googlevideo.com',
    'ytimg.com',
    'gvt1.com',
    'gvt2.com',

    # 'instagram.com',
    # 'facebook.com',
    # 'whatsapp.net',
    # 'cdninstagram.com',
    # 'fbcdn.net',
    # 'facebook.net',
    # 'fbsbx.com'
])


class MyBN(nn.BatchNorm1d) :
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device=None, dtype=None) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, L, x = input.shape
        input = input.view(N * L, x)
        return super().forward(input).view(N, L, x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 3600 * 2):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            idx: Tensor, shape ``[batch_size, seq_len]``
        """
        idx = torch.clip(idx, 0, self.max_len - 1).long()
        return self.pe[idx]

class DPIModel(nn.Module) :
    def __init__(self, n, num_category_inputs, num_cat_per_category_input, dpi_bytes) -> None:
        super().__init__()
        self.n = n
        self.n_embd_per_bytes = 8
        ks2 = max(dpi_bytes // 10 - 1, 1)
        self.dpi_bytes = dpi_bytes
        self.dpi_proj = nn.Sequential(
            nn.Conv1d(self.n_embd_per_bytes, 512, kernel_size = 20, stride = 10),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size = ks2, stride = 1), # 7, 15, 23, 31
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size = 1, stride = 1),
            # nn.Conv1d(512, 512, kernel_size = 8, stride = 2), # TODO
        )
        self.bytes_embd = nn.Embedding(256, self.n_embd_per_bytes)
        self.net = nn.Sequential(
            nn.Linear(n + 4 * num_category_inputs, 512),
            MyBN(512),
            nn.GELU(),
            nn.Linear(512, 512),
            MyBN(512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        self.pe = PositionalEncoding(256)
        self.host_embd = nn.Embedding(501, 512)
        self.num_category_inputs = num_category_inputs
        self.cat_embds = nn.ModuleList([nn.Embedding(num_cat_per_category_input, 4) for _ in range(num_category_inputs)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512) * 0.1, requires_grad = True)
        self.trans_block = nn.ModuleList()
        for i in range(2) :
            block = nn.TransformerEncoderLayer(512, 8, 2048, dropout = 0, activation = 'gelu', norm_first = True, batch_first = True)
            self.trans_block.append(block)
        self.out_layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x, xc, start_ts, end_ts, dpi_bytes, mask) :
        N, L, _ = x.shape
        assert xc.shape[2] == self.num_category_inputs
        dpi_embd = self.bytes_embd(dpi_bytes.long())
        embds = [self.cat_embds[i](xc[..., i].to(x.device)) for i in range(self.num_category_inputs)]
        embds = torch.cat(embds, dim = 2)
        x = torch.cat([x, embds], dim = 2)
        h: torch.Tensor = self.net(x).view(N, L, 512)

        dpi_embd = einops.rearrange(dpi_embd, 'N L B C -> (N L) C B')
        if self.dpi_bytes > 0 :
            h_dpi: torch.Tensor = self.dpi_proj(dpi_embd)
            h_dpi = einops.rearrange(h_dpi, '(N L) C 1 -> N L C', L = L)
            h = h + h_dpi

        pe_start_ts = self.pe(start_ts)
        pe_end_ts = self.pe(end_ts)
        pe = torch.cat([pe_start_ts, pe_end_ts], dim = -1)
        h = h + pe

        mask_padding = torch.ones(N, 1, dtype = torch.bool).to(mask.device)
        mask = torch.cat([mask_padding, mask], dim = 1)
        h = torch.cat([self.cls_token.repeat(N, 1, 1), h], dim = 1)
        for block in self.trans_block :
            h = block(h, src_key_padding_mask = mask)
        h = h[:, 0, :]
        val = self.out_layers(h)
        return val

