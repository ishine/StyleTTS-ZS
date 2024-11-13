import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm
from normalizations import *
from resblock import ResBlk
import math

class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


class TVStyleEncoder(nn.Module):
    def __init__(self, pre_out=512, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()        
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for i in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            if i % 2 == 0:
                blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            else:
                blocks += [ResBlk(dim_in, dim_out, downsample='timepreserve')]
            dim_in = dim_out
        
        
        blocks += [nn.LeakyReLU(0.2)]
#         blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
#         blocks += [nn.AdaptiveAvgPool2d((1, None))]
        self.to_q = spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))
        self.to_k = spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))
        self.to_v = spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))

        self.scale = dim_out ** -0.5
        
        self.shared = nn.Sequential(*blocks)
        
        self.out = nn.Linear(dim_out, style_dim)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):        
        h = self.shared(x)
        q = self.to_q(h).squeeze(-2)
        k = self.to_k(h).squeeze(-2)
        h = self.to_v(h).squeeze(-2)
        
        # compute attention
        batch_size = q.shape[0]
        time = q.shape[-1]
        sim = (k.transpose(-1, -2).reshape(batch_size * time, -1) * 
               q.transpose(-1, -2).reshape(batch_size * time, -1)).sum(axis=-1)
        sim = sim.reshape(batch_size, time) * self.scale
        attn = sim.softmax(dim=-1).unsqueeze(1)
                
        o = self.relu(attn @ h.transpose(-1, -2))
        o = o.view(o.size(0), -1)
        out = self.out(o)
        
        return out


class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, 
                                 d_model // 2, 
                                 num_layers=1, 
                                 batch_first=True, 
                                 bidirectional=True, 
                                 dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        
        
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
                
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)
        
        return x.transpose(-1, -2)
    
    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask