from typing import Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, o, g = torch.chunk(self.gates(torch.cat([x, h], dim=1)), 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class TemporalMemoryModule(nn.Module):
    """ConvLSTM temporal memory over aligned video frames."""

    def __init__(self, input_channels: int = 3, hidden_dim: int = 64, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, aligned_frames: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _, height, width = aligned_frames.shape
        h = aligned_frames.new_zeros(batch, self.hidden_dim, height, width)
        c = aligned_frames.new_zeros(batch, self.hidden_dim, height, width)
        for tidx in range(seq_len):
            h, c = self.cell(self.stem(aligned_frames[:, tidx]), (h, c))
        return self.out(h)
