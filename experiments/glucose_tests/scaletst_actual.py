from typing import List, Tuple, Optional, Union, Dict
from neuralforecast.losses.pytorch import BasePointLoss, DistributionLoss
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_dct as dct


from neuralforecast.common._base_model import BaseModel
from neuralforecast.common._modules import RevIN
from neuralforecast.losses.pytorch import MAE
from vqvae import train_and_get_VQVAE, train_and_get_VQVAEConv
# import helpers from your PatchTST implementation; adjust path if needed
from neuralforecast.models.patchtst import positional_encoding, TSTEncoder
import torch
from x_transformers import XTransformer, TransformerWrapper, Decoder
# --- Helper function for top-k filtering ---
def top_k_logits(logits, k):
    """
    Filters logits to only the top k values, setting others to -infinity.
    This is used for top-k sampling.
    """
    if k == 0:
        return logits
    v, ix = torch.topk(logits, k)
    out = torch.ones_like(logits) * float('-inf')
    out.scatter_(1, ix, v)
    return out
class ScaleTSTTokenizer(nn.Module):
    def __init__(self, patch_length, embed_dim):
        super(ScaleTSTTokenizer, self).__init__()
    def get_tokenized_embed(self, original_embed):
        pass
    def get_untokenized_embed(self, embed):
        pass
class SimpleScaleTokenizer(ScaleTSTTokenizer):
     def __init__(self, patch_length, embed_dim):
        super().__init__(patch_length, embed_dim)
        self.embed=nn.Linear(patch_length, embed_dim)
        self.unembed=nn.Linear(embed_dim, patch_length)
     def get_tokenized_embed(self, original_embed):
        return self.embed(original_embed)
     def get_untokenized_embed(self, embedded):
        return self.unembed(embedded) 
class VQVAETokenizer(ScaleTSTTokenizer):
    def __init__(self, patch_length, embed_dim, conv=False, codebook_size=512):
        super().__init__(patch_length, embed_dim)
        self.patch_length=patch_length
        self.embed_dim=embed_dim
        self.to_hidden=nn.Linear(patch_length, embed_dim)
        self.codebook_size=512
        self.conv=conv
    def train_vqvae(self, data):
        if(self.conv):
            self.model=train_and_get_VQVAEConv(self.patch_length, data.size()[1], self.embed_dim, self.codebook_size, data.to(torch.float32))
        else:
            self.model=train_and_get_VQVAE(self.patch_length, self.embed_dim, self.codebook_size, data.to(torch.float32))
        self.model.eval()
    def get_tokenized_embed(self, original_embed):
        original_embed=dct.dct(original_embed)
        with torch.no_grad():
            return self.model.get_quantized(original_embed)
    def get_untokenized_embed(
        self, 
        embedded, 
        temperature=1.0, 
        top_k=50
    ):
        """
        Converts LM output embeddings back to time-series patches by
        SAMPLING from the codebook.

        Args:
            embedded (torch.Tensor): Output from LM, shape (B, N, codebook_dim)
            temperature (float): Controls randomness. Higher is more random.
            top_k (int): Samples from the top k most likely codes. 0 to disable.
        """
        B, N, D = embedded.shape
        # (B, N, D) -> (B*N, D)
        embedded_flat = embedded.reshape(B * N, D)

        # Get the internal codebook
        # `self.model.vq._codebook.embed` has shape (heads, codebook_size, codebook_dim)
        # We assume heads=1 here.
        codebook = self.model.vq._codebook.embed.squeeze(0) # (codebook_size, codebook_dim)

        with torch.no_grad():
            self.model.eval()
            
            # Manually calculate distances to get logits
            if self.model.vq.use_cosine_sim:
                # Cosine Similarity
                embedded_flat_norm = F.normalize(embedded_flat, dim=-1)
                codebook_norm = F.normalize(codebook, dim=-1)
                logits = torch.matmul(embedded_flat_norm, codebook_norm.t()) # (B*N, codebook_size)
            else:
                # Negative Squared Euclidean Distance
                d_squared = torch.sum(embedded_flat**2, dim=-1, keepdim=True) - \
                            2 * torch.matmul(embedded_flat, codebook.t()) + \
                            torch.sum(codebook**2, dim=-1, keepdim=True).t()
                logits = -d_squared # (B*N, codebook_size)

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                logits = top_k_logits(logits, top_k)

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample indices from the distribution
            indices = torch.multinomial(probs, num_samples=1) # (B*N, 1)
           # print(indices)
            
            # Reshape indices back to (B, N)
            indices = indices.reshape(B, N)

        # --- This is the corrected decoding path ---
        
        # 1. Get the latent vectors (in embed_dim) corresponding to the sampled indices
        #    This projects from codebook_dim -> embed_dim
        quantized_vectors = self.model.vq.get_output_from_indices(indices)
        
        # 2. Decode the latent vectors back into patches
        #    This is the crucial step that uses the VQ-VAE's decoder
        return self.model.get_actual(quantized_vectors)
    # def get_untokenized_embed(self, embedded):
    #     embedded_=embedded.flatten(0,1).unsqueeze(0)
    #     with torch.no_grad():
    #         self.model.vq._codebook.eval()
    #         _, indices, _=self.model.vq._codebook(embedded_, freeze_codebook=True)
        

    #     return self.model.decoder(self.model.vq.get_output_from_indices(indices.reshape(embedded.size()[0], embedded.size()[1]))) 
class IdentityEmbed():
    def __call__(self, x, **kwargs):
        return x
class ScaleTST(BaseModel):
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )
    def __init__(
        self,
        h: int,
        input_size: int,
        patch_sizes: List[int],
        valid_batch_size: int = 32,
        inference_windows_batch_size: int=32,
        windows_batch_size: int = 32,
        max_steps: int = 1000,
        val_check_steps: int = 10,
        loss: BasePointLoss = MAE(),
        valid_loss: Union[BasePointLoss, DistributionLoss, nn.Module] = MAE(),
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        start_padding_enabled: bool = False,
        training_data_availability_threshold: Union[float, List[float]] = 0.0,
        step_size: int = 1,
        num_lr_decays: int = 0,
        early_stop_patience_steps: int = -1,
        scaler_type: str = "identity",
        futr_exog_list: Union[List, None] = None,
        hist_exog_list: Union[List, None] = None,
        stat_exog_list: Union[List, None] = None,
        conv_reduce: bool = False,
        tokenizer_class = SimpleScaleTokenizer,
        embed_dim: int = 512,
        transformer_depth: int=6,
        n_heads: int = 8,
        exclude_insample_y: Union[bool, None] = False,
        drop_last_loader: Union[bool, None] = False,
        random_seed: Union[int, None] = 1,
        alias: Union[str, None] = None,
        optimizer: Union[torch.optim.Optimizer, None] = None,
        optimizer_kwargs: Union[Dict, None] = None,
        lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
        lr_scheduler_kwargs: Union[Dict, None] = None,
        dataloader_kwargs=None,
        **trainer_kwargs):
        super(ScaleTST, self).__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
        )
        self.tokenizers = nn.ModuleList([
        tokenizer_class(patch_size, embed_dim, conv=True) for patch_size in patch_sizes
        ])
        self.revin_module=RevIN(1, affine=False)
        self.reducers=[]
        self.conv_reduce=conv_reduce
        self.is_vqvae=(tokenizer_class==VQVAETokenizer)
        self.small_patch=patch_sizes[-1]
        if(self.h % self.small_patch !=0):
            raise Exception("Smallest patch size must divide forecasting horizon!");
        self.h_default=h // self.small_patch
        self.patch_sizes=patch_sizes
        self.forward_length=h
        self.break_token=torch.randn(embed_dim).to('cuda')
        # for patch_size in patch_sizes:
        #     self.tokenizers.append(tokenizer_class(patch_size, embed_dim))
        #     if(self.conv_reduce):
        #         self.reducers.append(nn.Sequential(nn.Conv1d(embed_dim, embed_dim, patch_size), nn.MaxPool1d(patch_size)))
        self.transformer=TransformerWrapper(
            num_tokens=10000,
            max_seq_len=1024,
            token_emb=IdentityEmbed(),
            attn_layers=Decoder(
                dim=embed_dim,
                depth=transformer_depth,
                heads=n_heads,
                attn_flash=True
            )
        )
    def train_vqvae(self, training_data):
        for i in range(len(self.tokenizers)):
            print(i)
            print(self.patch_sizes[i])
            print(training_data.shape)
            training_data_base=training_data[(training_data.shape[0]%self.patch_sizes[i]):].unsqueeze(-1)
            training_data_=training_data_base.reshape(-1, self.patch_sizes[i])
            for j in range(1, self.patch_sizes[i]):
                training_data_=torch.cat((training_data_, torch.cat((training_data_base[j:], training_data_base[:j]), dim=0).reshape(-1, self.patch_sizes[i])), dim=0)
            print("Passing in size: {}".format(training_data_.size()))
            self.tokenizers[i].train_vqvae(training_data_)
    def train_vqvae_conv(self, training_data):
        for i in range(len(self.tokenizers)):
            print(i)
            print(self.patch_sizes[i])
            print(training_data.shape)
            training_data_base=training_data[(training_data.shape[0]%self.patch_sizes[i]):].unsqueeze(-1)
            training_data_=training_data_base.reshape(-1, 1)
            print("Passing in size: {}".format(training_data_.size()))
            self.tokenizers[i].train_vqvae(training_data_)
        
    # def train_vqvae(self, training_data):
    #     # training_data is 1D (L,)
        
    #     # 1. Reshape data to (batch_size, num_channels, length) for RevIN
    #     # We assume batch_size=1, num_channels=1
    #     # (L,) -> (1, 1, L)
    #     training_data_for_norm = training_data.unsqueeze(0).unsqueeze(0)

    #     # 2. Apply RevIN normalization
    #     # This will use the "norm" operation of your self.revin_module
    #     normalized_data_tensor = self.revin_module(training_data_for_norm, "norm")

    #     # 3. Squeeze data back to 1D (L,) for patching
    #     normalized_data = normalized_data_tensor.squeeze(0).squeeze(0)

    #     # 4. Proceed with your original logic, using the normalized_data
    #     for i in range(len(self.tokenizers)):
    #         print(i)
    #         print(self.patch_sizes[i])
    #         print(normalized_data.shape) # Shape of the normalized data
            
    #         patch_size = self.patch_sizes[i]
            
    #         # Use the normalized_data here
    #         training_data_ = normalized_data[(normalized_data.shape[0] % patch_size):].unsqueeze(-1).reshape(-1, patch_size)
            
    #         self.tokenizers[i].train_vqvae(training_data_)
    def forecast_step(self, index, embeddings, **kwargs):
        # Assumes embedding size of N, L, H
        patch_size=self.patch_sizes[index]
        extra_patches=math.ceil(self.forward_length/patch_size)
        for i in range(extra_patches):
            output=self.transformer(embeddings, return_embeddings=True, **kwargs)
            embeddings=output
        return embeddings
    
    def forward(self, windows_batch, h=None):
        if(h==None):
            h=self.h_default
        #Assumes X is of shape (N, L)
        #x = self.revin_module(windows_batch["insample_y"], "norm")
        x = windows_batch["insample_y"].squeeze(-1)
        embeddings=None
        for i in range(len(self.patch_sizes)):
            patch_size=self.patch_sizes[i]
            #Shape should be N, l=L/P, P
            if(not self.is_vqvae):
                X=x.reshape(x.shape[0], -1, patch_size)
            else:
                X=x.reshape(x.shape[0], -1, 1)                
            #Shape should be N, l, H
            X_embed=self.tokenizers[i].get_tokenized_embed(X)
            if(embeddings==None):
            #Shape should be N, l, H
                embeddings=X_embed
            else:
              #  break_token_repeat=self.break_token.unsqueeze(0).unsqueeze(0).repeat(X_embed.shape[0], 1, 1)
            #Shape should be N, l_1, H + N, l, H -> N, l+l_1, H
                embeddings=torch.cat((embeddings, X_embed), dim=1)
            #SHape should be N, l, H
            embeddings=self.forecast_step(i, embeddings)
        h_=h
        if(self.is_vqvae):
            h_*=self.patch_sizes[-1]
        pred_timesteps=self.tokenizers[-1].get_untokenized_embed(embeddings[:, -h_:]).flatten(-2, -1)
       # print("super cool size: {}".format(self.tokenizers[-1].get_untokenized_embed(embeddings[:, -h:]).size()))
        #return self.revin_module(pred_timesteps.reshape(x.size()[0], self.h, 1), "denorm").reshape(x.size()[0], self.h, self.loss.outputsize_multiplier)
        return pred_timesteps.reshape(x.size()[0], self.h, self.loss.outputsize_multiplier)