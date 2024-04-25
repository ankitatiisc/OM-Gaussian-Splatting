
import torch
import torch.nn as nn
import numpy as np
from scene.vqvae_modules.encoder import Encoder
from scene.vqvae_modules.quantizer import VectorQuantizer
from scene.vqvae_modules.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(12, dim_out, dim_hidden, num_layers)

        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return embedding_loss, z_q, perplexity

