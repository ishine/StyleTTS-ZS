import torch
import torch.nn as nn
from torch import Tensor, einsum
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from math import floor, log, pi
from Utility.utils import *

"""
Attention Components
"""


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor, num_buckets: int, max_distance: int
    ):
        num_buckets //= 2
        ret = (relative_position >= 0).to(torch.long) * num_buckets
        n = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, num_queries: int, num_keys: int) -> Tensor:
        i, j, device = num_queries, num_keys, self.relative_attention_bias.weight.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")

        relative_position_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )

        bias = self.relative_attention_bias(relative_position_bucket)
        bias = rearrange(bias, "m n h -> 1 h m n")
        return bias


class AttentionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context) + x
        return x
    
    
class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        use_rel_pos: bool,
        out_features: Optional[int] = None,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        mid_features = head_features * num_heads

        if use_rel_pos:
            assert exists(rel_pos_num_buckets) and exists(rel_pos_max_distance)
            self.rel_pos = RelativePositionBias(
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                num_heads=num_heads,
            )
        if out_features is None:
            out_features = features
            
        self.to_out = nn.Linear(in_features=mid_features, out_features=out_features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        # Compute similarity matrix
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = (sim + self.rel_pos(*sim.shape[-2:])) if self.use_rel_pos else sim
        sim = sim * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        # Compute and return attention
        return self.attention(q, k, v)

    
class CrossAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        num_time: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_key = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_k = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_v = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        
        self.num_time = num_time
        self.positions = nn.Embedding(num_time, features)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        idx = torch.arange(0, self.num_time).to(x.device)
        keys = self.positions(idx).transpose(-1, -2).expand(x.shape[0], -1, -1).transpose(-1, -2)
        
        # Normalize then compute q from input and k,v from context
        x, keys, context = self.norm(x), self.norm(keys), self.norm_context(context)
        q, k, v = (self.to_q(x), self.to_k(keys), self.to_v(context))

        # Compute and return attention
        return self.attention(q, k, v)