"""
VAE Model for Drawing Data
Extracted from vae_train.ipynb for easy importing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


class VAE64(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE64, self).__init__()
        
        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.GELU()
        )
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder: 1x1 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 128, 8, stride=1, padding=0), # 8x8
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # 64x64
            nn.Sigmoid() 
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --- Adjusted Loss Function ---
def weighted_vae_loss(recon_x, x, mu, logvar, epoch):
    # 1. Create the weight mask
    # We give the white pixels (the lines) 50x more weight than the background
    weight = torch.ones_like(x)
    weight[x > 0.5] = 50.0 
    
    # 2. Weighted MSE
    # (recon_x - x)^2 * weight
    mse = torch.sum(weight * (recon_x - x) ** 2)
    
    # 3. KL Divergence (Extreme Beta Warmup)
    # Start Beta at 0 and let it crawl up very slowly after epoch 50
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta = 0 if epoch < 50 else min(0.0001, (epoch - 50) * 0.000002)
    
    return mse + (beta * kld)

class CLIPHandler:
    def __init__(self, device):
        self.device = device
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze CLIP weights (we only want to use it, not train it)
        for param in self.model.parameters():
            param.requires_grad = False

    def embed_text(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # We use the pooled output (final representation of the sentence)
        return outputs.pooler_output

