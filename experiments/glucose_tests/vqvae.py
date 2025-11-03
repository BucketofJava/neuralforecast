import torch
from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize, Sequential
from conv_utils import make_strided_conv_sequence_1d, make_transpose_conv_sequence_1d
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
# 1. Define your VQ-VAE (Encoder -> VQ -> Decoder)
# This is a simplified example
class VQVAE(nn.Module):
    def __init__(self, dim, hidden_dim, codebook_size):
        super().__init__()
        self.encoder = nn.Linear(dim, dim) # Your encoder can be much more complex
        
        self.vq = VectorQuantize(
            dim = dim,
            codebook_dim=hidden_dim,
            codebook_size = codebook_size,
            commitment_weight = 1.,
            threshold_ema_dead_code = 10,
            decay = 0.5 # EMA decay
        )
        
        self.decoder = nn.Linear(dim, dim) # Your decoder
        
    def forward(self, x):
        # x shape: (batch_size, sequence_len, dim)
        encoded = self.encoder(x)
        
        # The vq layer returns the quantized vectors, indices, and commitment loss
        quantized, indices, commit_loss = self.vq(encoded)
        print("Unique Indices: {}".format(indices.unique().numel()))
        # Decode the quantized vectors
        reconstructed = self.decoder(quantized)
        return reconstructed, commit_loss
class VQVAE_Conv(nn.Module):
    def __init__(self, downsample_factor, dim, hidden_dim, codebook_size):
        super().__init__()
        self.encoder = make_strided_conv_sequence_1d(in_channels=1, out_channels=1, downsample_factor=downsample_factor)
        
        self.vq = VectorQuantize(
            dim = dim*downsample_factor,
            codebook_dim=hidden_dim,
            codebook_size = codebook_size,
            commitment_weight = 1.,
            threshold_ema_dead_code = 2,
            decay = 0.8 # EMA decay
        )
        
        self.decoder = make_transpose_conv_sequence_1d(in_channels=1, out_channels=1, upsample_factor=downsample_factor) # Your decoder
    def get_quantized(self, x):
        # x shape: (batch_size, sequence_len 1)
        print(f"X Size: {x.size()}")
        encoded = self.encoder(x.transpose(-1,-2)).transpose(-1,-2)
        print(f"Encoded Size: {encoded.size()}")
        # The vq layer returns the quantized vectors, indices, and commitment loss
        quantized, indices, commit_loss = self.vq(encoded)
        return quantized
    def get_actual(self, x):
        return self.decoder(x.transpose(-1,-2)).transpose(-1,-2)
    def forward(self, x, print_indices=False):
        # x shape: (batch_size, sequence_len 1)
        encoded = self.encoder(x.transpose(-1,-2)).transpose(-1,-2)

        
        # The vq layer returns the quantized vectors, indices, and commitment loss
        quantized, indices, commit_loss = self.vq(encoded)
        if(print_indices):
            print("Unique Indices: {}".format(indices.unique().numel()))
        # Decode the quantized vectors
        reconstructed = self.decoder(quantized.transpose(-1,-2)).transpose(-1,-2)
        return reconstructed, commit_loss

# --- Training Loop ---
def train_and_get_VQVAE(dim, hidden_dim, codebook_size, data):
    codebook_size = 4096
    model = VQVAE(dim, hidden_dim, codebook_size).to("cuda") # or .to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # A batch of your vectors
    vectors = data.to("cuda") # (batch, seq_len, dim)

    # 1. Set the model to TRAINING mode
    model.train()
    for i in range(1200):
    # 2. Forward pass
        reconstructed_vectors, commit_loss = model(vectors)
        
        # 3. Calculate loss
        reconstruction_loss = F.mse_loss(reconstructed_vectors, vectors)
        total_loss = reconstruction_loss + commit_loss
        print("VQVAE LOSS: {}".format(total_loss))
        print("Commit Loss: {}".format(commit_loss))
        # 4. Backpropagate and update
        total_loss.backward()
        opt.step()
        opt.zero_grad()
    return model
def train_and_get_VQVAEConv(dim, len, hidden_dim, codebook_size, data):
    codebook_size = 4096
    len=24
    model = VQVAE_Conv(len // dim, dim, hidden_dim, codebook_size).to("cuda") # or .to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # A batch of your vectors
    vectors = data.to("cuda").reshape(data.size()[0]//len, len, 1) # (batch, seq_len, dim)

    # 1. Set the model to TRAINING mode
    model.train()
    for i in range(1):
    # 2. Forward pass
        reconstructed_vectors, commit_loss = model(vectors)
        
        # 3. Calculate loss
        reconstruction_loss = F.mse_loss(reconstructed_vectors, vectors)
        total_loss = reconstruction_loss + commit_loss
        print("VQVAE LOSS: {}".format(total_loss))
        print("Commit Loss: {}".format(commit_loss))
        # 4. Backpropagate and update
        total_loss.backward()
        opt.step()
        opt.zero_grad()
    return model
# def train_and_get_VQVAEConv(dim, length, hidden_dim, codebook_size, data, batch_size=32, epochs=100, device="cuda"):
#     codebook_size = 4096
#     model = VQVAE_Conv(length // dim, dim, hidden_dim, codebook_size).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

#     # Create DataLoader for batching
#     dataset = TensorDataset(data)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#     model.train()
#     for epoch in range(epochs):
#         total_loss_epoch = 0.0
#         for (batch,) in dataloader:
#             batch = batch.to(device)  # (batch, seq_len, dim)
            
#             # Forward pass
#             reconstructed_vectors, commit_loss = model(batch, (epoch%20==0))

#             # Compute losses
#             reconstruction_loss = F.mse_loss(reconstructed_vectors, batch)
#             total_loss = reconstruction_loss + commit_loss

#             # Backpropagation
#             opt.zero_grad()
#             total_loss.backward()
#             opt.step()

#             total_loss_epoch += total_loss.item()

#         print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {total_loss_epoch / len(dataloader):.4f}")

#     return model

# When model.train() is active, the vq.forward() call
# will update its codebook via EMA automatically.