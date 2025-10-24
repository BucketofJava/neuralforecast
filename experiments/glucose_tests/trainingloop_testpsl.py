import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional

# Import necessary NeuralForecast components
from neuralforecast.auto import AutoNHITS, AutoNBEATS
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MSE

# Assume PSLoss_GDW and GDWLossMixin are importable
# These would typically reside in appropriate modules within the library
# For example: from neuralforecast.losses.pytorch import PSLoss_GDW
#              from neuralforecast.common._base_gdw import GDWLossMixin
# Since we don't have the actual files, we define them here for completeness
# but ideally, these would be imported.

# ==============================================================================
# PLACEHOLDER: Assume these classes exist and are importable
# In a real scenario, these imports would point to the actual modules
# ==============================================================================
# Attempt to import the real classes if they exist in the environment
from util_testing.patchwise_structural import PSLoss_GDW
from util_testing._patchstruct_mixin import GDWLossMixin # Hypothetical location
# except ImportError as e:
#     print(e)
#     print("Warning: PSLoss_GDW or GDWLossMixin not found in standard paths. Using placeholder definitions.")
#     # Define minimal placeholders if import fails, allowing script structure to work
#     from neuralforecast.losses.pytorch import BasePointLoss
#     class PSLoss_GDW(BasePointLoss): # Placeholder
#          def __init__(self, *args, **kwargs):
#               super().__init__(outputsize_multiplier=1, output_names=[""])
#               print("Warning: Using placeholder PSLoss_GDW.")
#          def __call__(self, y, y_hat, **kwargs):
#               loss = F.mse_loss(y_hat, y)
#               self.L_mse = loss.detach() # Mock storing value
#               self.L_corr = loss.detach()
#               self.L_var = loss.detach()
#               self.L_mean = loss.detach()
#               self.alpha_detached = torch.tensor(1.0)
#               self.beta_detached = torch.tensor(1.0)
#               self.gamma_detached = torch.tensor(1.0)
#               self.L_ps_weighted_detached = loss.detach()
#               return loss
#          def compute_gdw_total_loss(self, model_output_layer):
#               print("Warning: Using placeholder compute_gdw_total_loss.")
#               return self.L_mse * 2 # Mock calculation

#     class GDWLossMixin: # Placeholder
#         @property
#         def _gdw_output_layer(self):
#             common_names = ['output_projection', 'head', 'linear', 'c_out', 'flatten_and_linear']
#             for name in common_names:
#                 if hasattr(self, name):
#                     layer = getattr(self, name)
#                     if isinstance(layer, torch.nn.Module) and list(layer.parameters()):
#                         return layer
#             raise AttributeError("Placeholder GDWLossMixin could not find output layer.")
#         def backward(self, loss, *args, **kwargs):
#             print("Warning: Using placeholder GDWLossMixin backward.")
#             if isinstance(self.loss, PSLoss_GDW):
#                  output_layer = self._gdw_output_layer
#                  total_loss = self.loss.compute_gdw_total_loss(output_layer)
#                  super().backward(total_loss, *args, **kwargs)
#                  self.log('train_loss_total', total_loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
#             else:
#                  super().backward(loss, *args, **kwargs)

# ==============================================================================
# End of Placeholder Definitions
# ==============================================================================

import argparse
# Assuming glucose_experiment_configs.py is in the same directory or accessible
try:
    from glucose_experiment_configs import config_nbeats, config_nhits, init_optuna
except ImportError:
    print("Warning: glucose_experiment_configs.py not found. Using default configs.")
    # Define dummy configs if import fails
    config_nbeats = None
    config_nhits = None
    def init_optuna(*args, **kwargs): pass

from utilsforecast.plotting import plot_series

# ==============================================================================
# 1. Define Wrapper Classes for Auto Models using the Mixin
# ==============================================================================
class AutoNHITS_GDW(GDWLossMixin, AutoNHITS):
    """ AutoNHITS with PSLoss_GDW support via GDWLossMixin. """
    def __init__(self, **kwargs):
        # Ensure the loss is PSLoss_GDW if provided, otherwise default might break mixin
        if 'loss' in kwargs and not isinstance(kwargs['loss'], PSLoss_GDW):
             print(f"Warning: Initializing AutoNHITS_GDW with loss type {type(kwargs['loss'])}. "
                   f"GDW backward step will only work if loss is PSLoss_GDW.")
        elif 'loss' not in kwargs:
             raise ValueError("AutoNHITS_GDW requires a PSLoss_GDW instance during initialization.")
        super().__init__(**kwargs)


class AutoNBEATS_GDW(GDWLossMixin, AutoNBEATS):
    """ AutoNBEATS with PSLoss_GDW support via GDWLossMixin. """
    def __init__(self, **kwargs):
        if 'loss' in kwargs and not isinstance(kwargs['loss'], PSLoss_GDW):
             print(f"Warning: Initializing AutoNBEATS_GDW with loss type {type(kwargs['loss'])}. "
                   f"GDW backward step will only work if loss is PSLoss_GDW.")
        elif 'loss' not in kwargs:
             raise ValueError("AutoNBEATS_GDW requires a PSLoss_GDW instance during initialization.")
        super().__init__(**kwargs)

# ==============================================================================
# 2. Main Script Logic (Modified from previous version)
# ==============================================================================

def main():
    args = parse_args()
    print("Starting main function with PS Loss comparison")
    try:
        init_optuna()
    except NameError:
        print("Skipping Optuna initialization.")


    # --- Configuration ---
    GLUCOSE_DATA_DIR = "./simglucose_exog_9_day_test.csv"
    SAMPLE_FREQUENCY = "5min" # Corresponds to data frequency
    FORECAST_HORIZON = args.horizon # e.g., 6 (30 min), 12 (1 hour)
    INPUT_SIZE_ = args.input_size   # e.g., 72 (6 hours), 144 (12 hours) # Renamed to avoid global clash
    N_TIME_CV = 5 # Number of cross-validation folds
    TEST_SIZE = 288 # 1 day for 5min freq
    NUM_SAMPLES_OPTUNA = args.num_samples # Number of Optuna trials
    TARGET_PATIENT_ID="ALL"
    print(f"Horizon: {FORECAST_HORIZON}, Input Size: {INPUT_SIZE_}, Patient: {TARGET_PATIENT_ID}")

    # --- Load and Prepare Data ---
    try:
        glucose_df = pd.read_csv(GLUCOSE_DATA_DIR)
        if glucose_df.ds.dtype != '<M8[ns]':
            glucose_df['ds'] = pd.to_datetime(glucose_df['ds'], format='%Y-%m-%d %H:%M:%S')
    except FileNotFoundError:
        print(f"Error: Data file not found at {GLUCOSE_DATA_DIR}")
        return

    # Filter for the target patient
    glucose_df = glucose_df
    if glucose_df.empty:
        print(f"Error: No data found for patient ID '{TARGET_PATIENT_ID}'")
        return
    print(f"Data loaded and filtered for {TARGET_PATIENT_ID}. Shape: {glucose_df.shape}")

    # --- Instantiate Loss and Models ---
    # Use placeholder if real one wasn't imported
    ps_loss_instance = PSLoss_GDW(lambda_ps=3.0, delta_patch=36)

    models = [
        AutoNHITS(h=FORECAST_HORIZON, backend="optuna", config=config_nhits, loss=MSE()), # Original NHITS with MSE
        AutoNBEATS(h=FORECAST_HORIZON, backend="optuna", config=config_nbeats, loss=MSE()),# Original NBEATS with MSE
        AutoNHITS_GDW(h=FORECAST_HORIZON, backend="optuna", config=config_nhits, loss=ps_loss_instance), # NHITS with PS Loss
        AutoNBEATS_GDW(h=FORECAST_HORIZON, backend="optuna", config=config_nbeats, loss=ps_loss_instance) # NBEATS with PS Loss
    ]

    # --- Initialize NeuralForecast ---
    nf = NeuralForecast(models=models, freq=SAMPLE_FREQUENCY)
    gpu_trainer_kwargs = {
        "accelerator": "gpu",  # Tell Lightning to use the GPU
        "devices": 1,          # Use 1 GPU (set to -1 to use all available)
        "precision": "16-mixed", # Use 16-bit mixed precision for a large speedup
    }
    # --- Cross Validation ---
    print("Starting cross-validation...")
    try:
        cross_validation_df = nf.cross_validation(
            df=glucose_df,
            n_windows=N_TIME_CV,
            step_size=TEST_SIZE# Rolling window of 1 day
        )
        print("Cross-validation finished.")
        print("Cross-validation results sample:")
        print(cross_validation_df.head())
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        # Optionally, print more details or re-raise
        # import traceback
        # traceback.print_exc()
        return # Stop execution if CV fails

    # --- Evaluate ---
    print("\n--- Evaluation Metrics (Average MAE from CV) ---")
    model_names = [model.alias for model in models] # Get alias from instantiated models
    metrics = {name: [] for name in model_names}

    if not cross_validation_df.empty:
        for model_name in model_names:
             # Calculate MAE: abs(y - y_hat)
            mae_col = f'MAE_{model_name}'
            if model_name in cross_validation_df.columns:
                cross_validation_df[mae_col] = np.abs(cross_validation_df['y'] - cross_validation_df[model_name])
                # Avg across cutoffs for the single unique_id
                avg_mae = cross_validation_df[mae_col].mean()
                metrics[model_name] = avg_mae
                print(f"{model_name}: {avg_mae:.4f}")
            else:
                print(f"Warning: Model '{model_name}' not found in cross-validation results.")
                metrics[model_name] = np.nan
    cross_validation_df.to_csv("results_patchloss.csv")
    # --- Plotting ---
    if not cross_validation_df.empty:
        print("\nPlotting results from the first CV window...")
        plot_cv_results(glucose_df, cross_validation_df, INPUT_SIZE_)
        print("Plot saved to forecast_cv_plot.png")
    else:
        print("Skipping plotting due to empty cross-validation results.")

    print("Script finished.")


def plot_cv_results(original_df, forecast_df, plot_input_size):
    """
    Plots the ground truth and forecasts for the first cross-validation window.
    """
    models_to_plot = [col for col in forecast_df.columns if col not in ['unique_id', 'ds', 'cutoff', 'y'] and not col.startswith('MAE_')]
    plot_id = forecast_df['unique_id'].unique()[0]
    first_cutoff = forecast_df['cutoff'].min()
    forecast_subset = forecast_df[(forecast_df['unique_id'] == plot_id) & (forecast_df['cutoff'] == first_cutoff)].copy()
    actual_data_plot = forecast_subset[['unique_id', 'ds', 'y']].copy()
    plot_df = actual_data_plot.merge(forecast_subset[['unique_id', 'ds'] + models_to_plot], on=['unique_id', 'ds'], how='left')

    fig, ax = plt.subplots(figsize=(14, 7))
    try:
        # Use utilsforecast plotting
        plot_series(actual_data_plot, plot_df, models=models_to_plot, ax=ax, max_insample_length=plot_input_size)
    except Exception as e:
         print(f"Plotting error with utilsforecast: {e}. Falling back to basic plot.")
         # Fallback basic plot
         ax.plot(plot_df['ds'], plot_df['y'], label='Actual', color='black')
         for model in models_to_plot:
             if model in plot_df.columns:
                 ax.plot(plot_df['ds'], plot_df[model], label=model, linestyle='--')

    ax.set_title(f'Forecast vs Actuals for {plot_id} (First CV Window)')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Glucose Level')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./forecast_cv_plot.png')
    # plt.show()


def parse_args():
    desc = "Train and evaluate AutoNHITS/AutoNBEATS with and without PS Loss on Glucose data."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results (not used)')
    parser.add_argument('--horizon', type=int, default=6, help='Forecast horizon (e.g., 6 for 30min)')
    parser.add_argument('--input_size', type=int, default=72, help='Input size (e.g., 72 for 6hrs)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('--experiment_id', default='glucose_psloss_comparison', required=False, help='Optuna study name')
    parser.add_argument('--storage_name', default='sqlite:///optuna_study.db', required=False, help='Optuna storage URL')
    return parser.parse_args()

if __name__ == "__main__":
    print("Cuda available: {}".format(torch.cuda.is_available()))
    main()

