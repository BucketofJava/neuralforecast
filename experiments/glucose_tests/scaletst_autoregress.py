from typing import List, Tuple, Optional, Union, Dict
from neuralforecast.losses.pytorch import BasePointLoss, DistributionLoss
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_dct as dct
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, PatchTST  # The model to compare against
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import MAE, BasePointLoss
from neuralforecast.common._base_model import BaseModel
from neuralforecast.common._modules import RevIN
from neuralforecast.losses.pytorch import MAE
from vqvae import train_and_get_VQVAE, train_and_get_VQVAEConv
from neuralforecast.models.patchtst import positional_encoding, TSTEncoder
import torch
from x_transformers import XTransformer, TransformerWrapper, Decoder, AutoregressiveWrapper
from utilsforecast.data import generate_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae
from utilsforecast.plotting import plot_series
#Values: N - batch size. L - Sequence Length. F - Future Horizon size. P - Patch Size. H - Hidden Size.
#So we can just have model output its own loss and be fine
class IdentityLoss(BasePointLoss):
    def __init__(self, horizon_weight=None):
        super(IdentityLoss, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )
    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
        y_insample: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        return y_hat
#Tokenization:
#Define tokenizer abstract class
class IndexTokenizer():
    def __init__(self, patch_size, **kwargs):
        #super(IndexTokenizer, self).__init__()
        self.patch_size=patch_size
        self.codebook_size=0
    def get_indices(self, patched_data):
        pass
    def indices_to_values(self, indices):
        pass
#Define VQVAE
class VQVAETokenizer(IndexTokenizer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(patch_size, **kwargs)
        self.codebook_size=150
    def train_vqvae(self, data):
        self.model=train_and_get_VQVAE(self.patch_size, 256, self.codebook_size, data.to(torch.float32))
        self.model.eval()
    def get_indices(self, patched_data, index=0):
        # Assumed shape of patches: (N, L//P, P)
        tokenized_indices=self.model.get_quantized(patched_data)[1]
        tokenized_indices+=index*self.codebook_size
        return tokenized_indices
    def indices_to_values(self, indices, index):
        real_vals=self.model.get_actual_from_indices(indices%150)
      #  print(f"Real vals size: {real_vals.size()}")
        return real_vals
#Define a simple tokenizer that just takes the mean of the values in a patch and tokenizes as 1 0 -1s
class MeanSignTokenizer(IndexTokenizer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(patch_size, **kwargs)
        self.codebook_size=3
    def get_indices(self, patched_data, index=0):
        # Assumed shape of patches: (N, L//P, P)
        means=torch.sign(patched_data.mean(dim=-1))+1
        means+=index*self.codebook_size
    #    print(f"Size of means: {means.size()}")
        return means.long()
    def indices_to_values(self, indices):
        #Assumed shape of indices: (N, F//P)
        indices=(indices-2)
        #Should transform (N, F//P) -> (N, F)
        return indices.repeat(1, self.patch_size).float()

# Model Header: Include all parameters needed for BaseModel in addition to extras like patch_sizes
class AutoregressiveScaleModel(BaseModel):
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )
#Init method: call super init, store needed variables, initialize tokenizers and transformer
    def __init__(self,
                h,
                input_size,
                patch_sizes,
                transformer_depth,
                n_heads,
                embed_dim,
                tokenizer_class=MeanSignTokenizer,
                max_seq_len=1000,
                valid_batch_size: int = 32,
                inference_windows_batch_size: int=32,
                windows_batch_size: int = 32,
                max_steps: int = 1000,
                val_check_steps: int = 10,
                loss: BasePointLoss = IdentityLoss(),
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
                exclude_insample_y: Union[bool, None] = False,
                drop_last_loader: Union[bool, None] = False,
                random_seed: Union[int, None] = 1,
                alias: Union[str, None] = None,
                optimizer: Union[torch.optim.Optimizer, None] = None,
                optimizer_kwargs: Union[Dict, None] = None,
                lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
                lr_scheduler_kwargs: Union[Dict, None] = None,
                dataloader_kwargs=None,
                tokenizer_kwargs=None,
                **trainer_kwargs
                #TODO add the rest of the basemodel stuff
                ):
        super(AutoregressiveScaleModel, self).__init__(
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
        self.revin_module=RevIN(1, affine=False)
        self.h=h
        self.input_size=input_size
        self.tokenizer_class=tokenizer_class
        self.patch_sizes=patch_sizes
        self.patch_small=patch_sizes[-1]
        self.tokenizers=[(tokenizer_class(ps, **tokenizer_kwargs) if tokenizer_kwargs!=None else tokenizer_class(ps)) for ps in self.patch_sizes]
       # self.tokenizers=nn.ModuleList(self.tokenizer_list)
        # for tokenizer in self.tokenizers:
        #     for p in tokenizer.parameters():
        #         p.requires_grad_=False
        self.codebook_size=self.tokenizers[0].codebook_size
        num_tokens=int(sum([t.codebook_size for t in self.tokenizers])*1.2)
        self.transformer_model=TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
            dim=embed_dim,
            depth=transformer_depth,
            heads=n_heads,
            attn_flash=True
            )
        ).cuda()
#Wrap transformer in autoregressive wrapper
        self.transformer_model=AutoregressiveWrapper(self.transformer_model)
    #TODO? Doesn't seem necessary ATM
#Training tokenizer method. For now can just be VQVAE and a pass on the mean one. Important: take in patch index and add codebook_size*patch_index to indices to ensure unique indices!
    def train_tokenizer(self, training_data):
        if(self.tokenizer_class!=VQVAETokenizer):
            return
        for i in range(len(self.tokenizers)):
            training_data_base=training_data[(training_data.shape[0]%self.patch_sizes[i]):].unsqueeze(-1)
            training_data_=training_data_base.reshape(-1, self.patch_sizes[i])
            for j in range(1, self.patch_sizes[i]):
                training_data_=torch.cat((training_data_, torch.cat((training_data_base[j:], training_data_base[:j]), dim=0).reshape(-1, self.patch_sizes[i])), dim=0)
        #    print("Passing in size: {}".format(training_data_.size()))
            self.tokenizers[i].train_vqvae(training_data_)

#Forecasting step: Take in an index or patch size, as well as the current token sequence. Output the log probs for the next token in the sequence
#Repeat until stretched out over forecasting horizon with the patches
    def forecast_step(self, indices, patch_size, mask_index=1, return_loss=False):
        logits=self.transformer_model.net(indices[:, :-1])
        if(return_loss):
            target=indices[:, 1:].clone().long()
            target[:, :mask_index]=-100
            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                target,
                ignore_index=-100
            )
            return loss, logits[mask_index:]
        return logits[mask_index:]

#Core model loop: TRAINING
    def forward(self, windows_batch):
      #  print(f"Size of insample: {windows_batch['insample_y'].size()}")
        real_data=windows_batch['insample_y']
     #   real_data.requires_grad_=False
        cur_seq=None
        tot_loss=0
        for i in range(len(self.patch_sizes)):
       #     print(f"RD size: {real_data.size()}")
            real_data_=real_data.unsqueeze(-1).reshape((real_data.size()[0], -1,patch_sizes[i]))
          #  print(f"RD_ size: {real_data_.size()}")
            #Should return a sequence of longs
            tokenized=(self.tokenizers[i].get_indices(real_data_, i))
          #  print(f"T size: {tokenized.size()}")
            if(cur_seq==None):
                indices=tokenized
                cur_seq=indices
            else:
                indices=torch.cat((cur_seq, tokenized), dim=1)
           # indices=indices.detach()
            if(self.training):
                loss=self.forecast_step(indices, self.patch_sizes[i], mask_index=max(indices.shape[0]-tokenized.shape[0], 1), return_loss=True)[0]
                tot_loss=tot_loss+loss
            else:
                cur_seq=self.transformer_model.generate(indices, seq_len=self.h // self.patch_small)
            # cur_seq=cur_seq.detach()
        if(self.training):
            return tot_loss
     #   print(f"Sequence length: {cur_seq.size()[0]}")
        print(f"Indices: {cur_seq[:, -(self.h // self.patch_small):]}")
        Y=self.tokenizers[-1].indices_to_values(cur_seq[:, -(self.h // self.patch_small):], len(self.tokenizers)-1).flatten(-2, -1)
        #return self.revin_module(Y.unsqueeze(-1), "denorm").reshape(Y.size())
        return Y

                
            
#Revin everything
#Take in train, test data
#Loop through patch sizes
#Compute token indices for the i-th patch, and append to the current sequence
#Pass into the model 
#Get the output logits, run crossentropyloss with the 'true' tokens as determined by the real test data
#Append to the token sequence, detach everything from grad, and pass back in for the next batch
#Un-revin the final result for inference

#Core model loop: EVAL
#Just do the above but don't pass in shit
#Maybe add a training parameter to the above that runs it like normal and then otherwise doesn't wow

#Actual training step
#Overload pytorch lightning to run the model and pass in the optimizer and loss and do it w/i model!

if __name__ == "__main__":
    print("Testing the model")
    HORIZON=18
    input_size=36
    patch_sizes=[12, 6, 3]
    ar_model=AutoregressiveScaleModel(HORIZON, input_size, patch_sizes, 6, 8,128, max_steps=1000, alias="Model 1", tokenizer_class=VQVAETokenizer)
    ar_model.train()
    Y_df = generate_series(n_series=100, min_length=100, max_length=200)
    torch.set_float32_matmul_precision('medium')
    # Split into train and test
    Y_train_df = Y_df.groupby('unique_id').head(-HORIZON)
    Y_test_df = Y_df.groupby('unique_id').tail(HORIZON)
    print(f"Train length: {len(Y_train_df)}")
    print(f"Test length: {len(Y_test_df)} ")
    data_pt=torch.tensor(Y_train_df['y'].to_numpy())
    single_scale = AutoregressiveScaleModel(HORIZON, input_size, [6, 3], 6, 8,128, max_steps=1000, alias="Model 2", tokenizer_class=VQVAETokenizer)
    single_scale.train()
    
    ar_model.train_tokenizer(data_pt)
    single_scale.train_tokenizer(data_pt)
    nf = NeuralForecast(models=[ar_model, single_scale], freq='D')
    nf.fit(df=Y_train_df)
    ar_model.eval()
    predictions_df = nf.predict()
    print(f"Predictions Length: {len(predictions_df)}")
    predictions_df.to_csv("great_predictions_2.csv")
    Y_test_df.to_csv("great_tests_2.csv")
    Y_test_with_preds_df = Y_test_df[['ds', 'y']].merge(
        predictions_df, 
        on=['ds'], 
        how='left'
    )
    
    # Calculate MAE for each model
    # This dataframe has the MAE per unique_id
    evaluation_df = evaluate(
        Y_test_with_preds_df, 
        metrics=[mae],
        models=['AutoregressiveScaleModel'], # Aliases are class names
        target_col='y'
    )
    print(f"MAE for model: {np.sum(np.abs(predictions_df['Model 1'].to_numpy()-Y_test_df['y'].to_numpy()))/900}")
    print(f"MAE for ts model: {np.sum(np.abs(predictions_df['Model 2'].to_numpy()-Y_test_df['y'].to_numpy()))/900}")
    evaluation_df.to_csv("initial_ar_eval.csv")
