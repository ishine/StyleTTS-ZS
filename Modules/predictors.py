import torch
from torch import nn
from .encoders import *
from .adain_blocks import *


class DurationPredictor(nn.Module):
    def __init__(self, num_heads=8, head_features=64, d_hid=512, nlayers=6, max_dur=50):
        super().__init__()
        
        self.transformer = Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=nlayers,
             depthwise_conv_kernel_size=7,
             use_group_norm=True,
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
    def forward(self, text, style, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), text.size(1), max_size).to(text.device)
        
        text = text[..., :input_lengths.max()]
        
        mel_len = style.size(-1) # length of mel
        input_lengths = input_lengths + mel_len
        x = torch.cat([style, text], axis=-1).transpose(-1, -2) # last dimension
        
        x, _ = self.transformer(x, input_lengths)
        x = x.transpose(-1, -2)
        x_text = x[:, :, mel_len:]
        text_return[:, :, :x_text.size(-1)] = x_text        
        out = self.duration_proj(text_return.transpose(-1, -2))
        return out


class ProsodyPredictor(nn.Module):
    def __init__(self, num_heads=8, head_features=64, d_hid=512, nlayers=6, scale_factor=2):
        super().__init__()
        
        self.conf_pre = Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=nlayers // 2,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,
        )
                
        self.conf_after = Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=nlayers // 2,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,
        )
        
        self.F0_proj = LinearNorm(d_hid, 1)
        self.N_proj = LinearNorm(d_hid, 1)
        
        self.scale_factor = scale_factor
        
    def forward(self, text, style, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), text.size(1), max_size * self.scale_factor).to(text.device)
        
        text = text[..., :input_lengths.max()]
        
        mel_len = style.size(-1) # length of mel
        input_lengths = input_lengths + mel_len
        x = torch.cat([style, text], axis=-1).transpose(-1, -2) # last dimension
        
        x, _ = self.conf_pre(x, input_lengths)
        x = F.interpolate(x.transpose(-1, -2), scale_factor=self.scale_factor, mode='nearest').transpose(-1, -2)
        x, _ = self.conf_after(x, input_lengths * self.scale_factor)
        
        x = x.transpose(-1, -2)
        x_text = x[:, :, mel_len * self.scale_factor:]
        text_return[:, :, :x_text.size(-1)] = x_text  
        F0 = self.F0_proj(text_return.transpose(-1, -2)).squeeze(-1)
        N = self.N_proj(text_return.transpose(-1, -2)).squeeze(-1)
        return F0, N