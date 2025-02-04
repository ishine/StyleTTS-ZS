import torch.nn as nn
from torchaudio.models import Conformer
from .embeddings import *
from typing import Optional

import torch
from torch import Tensor, nn

from Utility.utils import *
from .sampler import *

"""
Diffusion Classes (generic for 1d data)
"""


class Model1d(nn.Module):
    def __init__(self, unet_type: str = "base", **kwargs):
        super().__init__()
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        self.unet = None
        self.diffusion = None

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(self, *args, **kwargs) -> Tensor:
        return self.diffusion.sample(*args, **kwargs)


"""
Audio Diffusion Classes (specific for 1d audio data)
"""


def get_default_model_kwargs():
    return dict(
        channels=128,
        patch_size=16,
        multipliers=[1, 2, 4, 4, 4, 4, 4],
        factors=[4, 4, 4, 2, 2, 2],
        num_blocks=[2, 2, 2, 2, 2, 2],
        attentions=[0, 0, 0, 1, 1, 1, 1],
        attention_heads=8,
        attention_features=64,
        attention_multiplier=2,
        attention_use_rel_pos=False,
        diffusion_type="v",
        diffusion_sigma_distribution=UniformDistribution(),
    )


def get_default_sampling_kwargs():
    return dict(sigma_schedule=LinearSchedule(), sampler=VSampler(), clamp=True)


class AudioDiffusionConditional(Model1d):
    def __init__(
        self,
        embedding_features: int,
        embedding_max_length: int,
        embedding_mask_proba: float = 0.1,
        **kwargs,
    ):
        self.embedding_mask_proba = embedding_mask_proba
        default_kwargs = dict(
            **get_default_model_kwargs(),
            unet_type="cfg",
            context_embedding_features=embedding_features,
            context_embedding_max_length=embedding_max_length,
        )
        super().__init__(**{**default_kwargs, **kwargs})

    def forward(self, *args, **kwargs):
        default_kwargs = dict(embedding_mask_proba=self.embedding_mask_proba)
        return super().forward(*args, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            **get_default_sampling_kwargs(),
            embedding_scale=5.0,
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})

class StyleDiffuser(nn.Module):
    def __init__(self, mel_dim=512, text_dim=768, style_dim=512, num_heads=8, num_layers=6, embedding_max_length=512):
        super().__init__()        
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        self.feature_proj = nn.Conv1d(514, text_dim, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList()
        
        self.blocks.append(Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,
        ))
        for _ in range(num_layers - 1):
            self.blocks.append(Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=7,
            use_group_norm=True,
        ))
        
        self.out = nn.Conv1d(text_dim, style_dim, 1, 1, 0)
        
        self.to_time = nn.Sequential(
                        TimePositionalEmbedding(
                            dim=text_dim, out_features=text_dim
                        ),
                        nn.GELU(),
                        nn.Linear(text_dim, text_dim),
                        nn.GELU(),
                        nn.Linear(text_dim, text_dim),
                        nn.GELU(),
                    )
        
        self.fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length, features=text_dim
        )
        self.fixed_feature = FixedEmbedding(
            max_length=embedding_max_length * 4, features=514
        )
        
        self.embedder = NumberEmbedder(features=text_dim)
        
        self.sep = nn.Embedding(num_embeddings=2, embedding_dim=text_dim)
        
    def run(self, x, time, embedding, features, input_lengths):
        
        mapping = self.to_time(time)
        
        mel = x

        text = embedding.transpose(-1, -2)
        
        text = text[..., :input_lengths.max()]
        
        mel = self.mel_proj(mel)
        pos = self.embedder(torch.arange(mel.shape[-1]).to(mel.device)).transpose(-1, -2).expand(mel.size(0), -1, -1)
        mel += pos
        
        features = self.feature_proj(features)
        
        feat_sep = self.sep(torch.zeros(x.size(0)).long().to(x.device)).unsqueeze(-1)
        
        text = torch.cat([features, feat_sep, text], axis=-1)
        
        mel_len = mel.size(-1) # length of mel
        
        input_lengths = input_lengths + mel_len + features.size(-1) + 2
        x_sep = self.sep(torch.ones(x.size(0)).long().to(x.device)).unsqueeze(-1)

        x = torch.cat([mel, x_sep, text], axis=-1).transpose(-1, -2) # last dimension
        mapping = mapping.unsqueeze(1).expand(-1, x.size(1), -1)
        
        for b in self.blocks:
            x = x + mapping
            x, input_lengths = b(x, input_lengths)
            
        x = x.transpose(-1, -2)
        x_mel = x[:, :, :mel_len]
        
        s = self.out(x_mel)        
        
        return s
    
    def forward(self, x: Tensor, 
                time: Tensor, 
                input_lengths: Tensor,
                embedding_mask_proba: float = 0.0,
                embedding: Optional[Tensor] = None, 
                features: Optional[Tensor] = None,
               embedding_scale: float = 1.0,
               feature_scale: float = 1.0,

               ) -> Tensor:
        
        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        fixed_features = self.fixed_feature(features.transpose(-1, -2)).transpose(-1, -2)

        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            
            features = torch.where(batch_mask, fixed_features, features)


        if embedding_scale != 1.0 or feature_scale != 1.0:
            # Compute both normal and fixed embedding outputs
            out = self.run(x=x, time=time, input_lengths=input_lengths, embedding=embedding, features=features)
#             out_masked = self.run(x=x, time=time, input_lengths=input_lengths, embedding=fixed_embedding, features=features)
            out_emb_masked = self.run(x=x, time=time, input_lengths=input_lengths, embedding=fixed_embedding, features=features)
            out_feat_masked = self.run(x=x, time=time, input_lengths=input_lengths, embedding=embedding, features=fixed_features)
            out_masked = self.run(x=x, time=time, input_lengths=input_lengths, embedding=fixed_embedding, features=fixed_features)

            if embedding_scale == 1.0:
                return out_feat_masked + (out - out_feat_masked) * feature_scale
            if feature_scale == 1.0:
                return out_emb_masked + (out - out_emb_masked) * embedding_scale

            return out_masked + (out_emb_masked - out_masked) * embedding_scale + (out_feat_masked - out_masked) * embedding_scale
        else:
            return self.run(x=x, time=time, input_lengths=input_lengths, embedding=embedding, features=features)
        
        return x