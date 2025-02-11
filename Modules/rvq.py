import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union
from einops import rearrange
from Modules.discriminators import WNConv1d


class VectorQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937
    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """

    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[2]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        """
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        encodings = rearrange(z_e, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=z.size(0))

        # vector quantization cost that trains the embedding vectors
        z_q = self.codebook(indices).transpose(1, 2)  # (B x D x T)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices

    def embed_code(self, embed_id):
        emb = F.embedding(embed_id, self.codebook.weight)
        return self.out_proj(emb.transpose(1, 2))
    
class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 256,
        n_codebooks: int = 7,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim)
                for i in range(n_codebooks)
            ]
        )

    def forward(self, z, n_quantizers: Union[None, int, Tensor] = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        codebook_indices = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        for i, quantizer in enumerate(self.quantizers):
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.size(0),), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)

        return z_q, commitment_loss, codebook_loss, torch.stack(codebook_indices, dim=1)

    def encode(self, z, n_quantizers: int = None):
        residual = z
        codes = []
        for quantizer in self.quantizers[:n_quantizers]:
            z_q_i, _, _, indices_i = quantizer(residual)
            residual = residual - z_q_i

            codes.append(indices_i)

        return torch.stack(codes, dim=-1)

    def decode(self, codes):
        z_q = 0

        for i, indices_i in enumerate(codes.unbind(dim=-1)):
            z_q += self.quantizers[i].embed_code(indices_i)

        return z_q